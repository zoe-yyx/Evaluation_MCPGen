"""
tasks/improved_prompts.py
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import ast
logger = logging.getLogger(__name__)


class ImprovedPromptBuilder:
    """改进的 Prompt 构建器"""
    
    def __init__(self, workflow_dir: Path):
        self.workflow_dir = workflow_dir
        self.workflow_context = self._load_workflow_context()
        self.server_context = self._load_server_context()
        self.dependencies_context = self._load_dependencies_context()
    
    def build_tool_generation_prompt(
        self,
        tool_name: str,
        original_tool_path: Optional[Path] = None,
        include_examples: bool = True
    ) -> str:
        """
        构建工具生成的 prompt
        
        Args:
            tool_name: 工具名称
            original_tool_path: 原始工具路径（用于提取签名）
            include_examples: 是否包含示例
        """
        # 1. 提取工具在工作流中的使用信息
        tool_usage = self._extract_tool_usage(tool_name)
        
        # 2. 提取原始工具签名（如果存在）
        original_signature = None
        if original_tool_path and original_tool_path.exists():
            original_signature = self._extract_tool_signature(
                original_tool_path, 
                tool_name  
            )
        
        # 3. 构建 prompt
        prompt_parts = [
            self._build_header(),
            self._build_context_section(tool_name, tool_usage),
            self._build_requirements_section(original_signature),
            self._build_constraints_section(),
        ]
            
        prompt_parts.append(self._build_output_instructions())
        
        return "\n\n".join(prompt_parts)
    
    def _load_workflow_context(self) -> Dict[str, Any]:
        """加载工作流上下文"""
        workflow_json = self.workflow_dir / "workflow.json"
        if not workflow_json.exists():
            return {}
        
        try:
            with open(workflow_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load workflow context: {e}")
            return {}
    
    def _load_server_context(self) -> Dict[str, Any]:
        """加载服务器上下文"""
        server_py = self.workflow_dir / "mcp_server" / "server.py"
        if not server_py.exists():
            return {}
        
        try:
            with open(server_py, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取 docstring
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            docstring = docstring_match.group(1).strip() if docstring_match else ""
            
            return {
                'description': docstring,
                'has_fastmcp': '@mcp.tool()' in content or 'from mcp' in content.lower(),
            }
        except Exception as e:
            logger.warning(f"Failed to load server context: {e}")
            return {}
    
    def _load_dependencies_context(self) -> Dict[str, Any]:
        """加载依赖上下文"""
        pyproject = self.workflow_dir / "pyproject.toml"
        if not pyproject.exists():
            return {}
        
        try:
            with open(pyproject, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单提取 dependencies
            deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if deps_match:
                deps_text = deps_match.group(1)
                # 清理引号
                deps = [
                    dep.strip().strip('"').strip("'")
                    for dep in deps_text.split(',')
                    if dep.strip()
                ]
                return {'dependencies': deps}
        except Exception as e:
            logger.warning(f"Failed to load dependencies: {e}")
        
        return {}
    
    def _extract_tool_usage(self, tool_name: str) -> Dict[str, Any]:
        usage_info = {
            'step_description': None,
            'parameters_example': None,
            'context_snippets': [],
        }

        if self.workflow_context:
            for step in self.workflow_context.get('workflow_steps', []):
                if step.get('mcp_tool') == tool_name:
                    usage_info['step_description'] = step.get('description')
                    usage_info['parameters_example'] = step.get('parameters')
                    break
        
        run_workflow = self.workflow_dir / "run_workflow.py"
        if run_workflow.exists():
            try:
                with open(run_workflow, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                pattern = rf'call_tool\(\s*["\']({tool_name})["\']'
                for match in re.finditer(pattern, content):
                    start = max(0, match.start() - 200)
                    end = min(len(content), match.end() + 200)
                    snippet = content[start:end]
                    usage_info['context_snippets'].append(snippet)
            except Exception as e:
                logger.warning(f"Failed to extract tool usage: {e}")
        
        return usage_info
    
    def _extract_tool_signature(self, tool_path: Path, tool_name: str) -> Optional[str]:
        """
        提取工具签名（支持 async def）
        
        Args:
            tool_path: 工具文件路径
            tool_name: 工具函数名（不是文件名！）
        
        Returns:
            完整的函数签名字符串
        """
        try:
            with open(tool_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # 🔥 修复：使用 tool_name 而不是 tool_path.stem
                    if node.name == tool_name:
                        # 构建签名字符串
                        is_async = isinstance(node, ast.AsyncFunctionDef)
                        func_prefix = "async def" if is_async else "def"
                        
                        # 提取参数
                        params = []
                        for arg in node.args.args:
                            param_str = arg.arg
                            if arg.annotation:
                                if hasattr(ast, 'unparse'):
                                    param_str += f": {ast.unparse(arg.annotation)}"
                                else:
                                    param_str += f": {ast.get_source_segment(content, arg.annotation) or 'Any'}"
                            params.append(param_str)
                        
                        params_str = ", ".join(params)
                        
                        # 提取返回类型
                        return_type = ""
                        if node.returns:
                            if hasattr(ast, 'unparse'):
                                return_type = f" -> {ast.unparse(node.returns)}"
                            else:
                                return_type = f" -> {ast.get_source_segment(content, node.returns) or 'Any'}"
                        
                        signature = f"{func_prefix} {node.name}({params_str}){return_type}"
                        logger.info(f"✓ Extracted signature: {signature}")
                        return signature
                    
                    # 🔥 可选：添加模糊匹配（像 ToolReplacer 一样）
                    if self._fuzzy_match_name(node.name, tool_name):
                        logger.info(f"✓ Found '{tool_name}' via fuzzy match as '{node.name}'")
                        # 构建签名字符串
                        is_async = isinstance(node, ast.AsyncFunctionDef)
                        func_prefix = "async def" if is_async else "def"
                        
                        # 提取参数
                        params = []
                        for arg in node.args.args:
                            param_str = arg.arg
                            if arg.annotation:
                                if hasattr(ast, 'unparse'):
                                    param_str += f": {ast.unparse(arg.annotation)}"
                                else:
                                    param_str += f": {ast.get_source_segment(content, arg.annotation) or 'Any'}"
                            params.append(param_str)
                        
                        params_str = ", ".join(params)
                        
                        # 提取返回类型
                        return_type = ""
                        if node.returns:
                            if hasattr(ast, 'unparse'):
                                return_type = f" -> {ast.unparse(node.returns)}"
                            else:
                                return_type = f" -> {ast.get_source_segment(content, node.returns) or 'Any'}"
                        
                        signature = f"{func_prefix} {node.name}({params_str}){return_type}"
                        logger.info(f"✓ Extracted signature: {signature}")
                        return signature
            
            logger.warning(f"✗ Function '{tool_name}' not found in {tool_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract signature: {e}")
            return None

    def _fuzzy_match_name(self, name1: str, name2: str) -> bool:
        """模糊匹配函数名（去掉下划线和 'tool' 后缀）"""
        def normalize(s):
            return s.replace('_', '').replace('tool', '').lower()
        return normalize(name1) == normalize(name2)

    def _build_header(self) -> str:
        return """# MCP Tool Generation Task

You are an expert Python developer tasked with generating a tool function for a FastMCP-based server.
Your goal is to create production-quality code that integrates seamlessly with the existing workflow."""
    
    def _build_context_section(
        self,
        tool_name: str,
        tool_usage: Dict[str, Any]
    ) -> str:
        """构建上下文部分"""
        sections = [
            "## Context",
            f"**Tool Name:** `{tool_name}`",
        ]
        
        # 工作流描述
        if self.workflow_context:
            workflow_info = self.workflow_context.get('workflow', {})
            if workflow_info.get('description'):
                sections.append(f"**Workflow Purpose:** {workflow_info['description']}")
        
        # 步骤描述
        if tool_usage.get('step_description'):
            sections.append(f"**Tool Purpose:** {tool_usage['step_description']}")
        
        # 服务器上下文
        if self.server_context.get('description'):
            sections.append(f"**Server Context:** {self.server_context['description']}")
        
        # 使用示例
        if tool_usage.get('parameters_example'):
            sections.append("**Usage Example:**")
            sections.append(f"```json\n{json.dumps(tool_usage['parameters_example'], indent=2)}\n```")
        
        # 可用依赖
        if self.dependencies_context.get('dependencies'):
            deps = self.dependencies_context['dependencies']
            sections.append(f"**Available Dependencies:** {', '.join(deps[:5])}" + 
                          (f" (and {len(deps) - 5} more)" if len(deps) > 5 else ""))
        
        return "\n".join(sections)
    
    def _build_requirements_section(
        self,
        original_signature: Optional[str]
    ) -> str:
        """构建需求部分"""
        requirements = [
            "## Requirements",
            "",
            "### CRITICAL Requirements (Must Follow):",
            "",
            "1. **Function Signature:**",
        ]
        
        if original_signature:
            requirements.extend([
                f"   - Must match this signature: `{original_signature}`",
                "   - Parameter names and types must be identical",
            ])
        else:
            requirements.extend([
                "   - Use descriptive parameter names",
                "   - All parameters must have type hints",
            ])
                
        return "\n".join(requirements)
    
    def _build_constraints_section(self) -> str:
        """构建约束部分"""
        return """## Constraints
**What NOT to do:**
- Do NOT use `*args` or `**kwargs`
- Do NOT use `@mcp.tool()` decorator (it's added by the server)
- Do NOT import unavailable modules
- Do NOT use `eval()`, `exec()`, or other dangerous operations
- Do NOT include test code or example usage in the file
- Do NOT add extra helper functions (keep it to just the one function)"""
    
    def _build_output_instructions(self) -> str:
        """构建输出指令"""
        return """## Output Instructions

Generate ONLY the Python function code. 

**The output should start with `def` and end with the last line of the function.**"""

