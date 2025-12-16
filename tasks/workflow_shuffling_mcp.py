"""
Unified Workflow Orchestration Evaluation System (MCP Version)
统一的工作流编排评测系统 - 通过MCP Server获取Tool信息

核心功能：
1. 通过运行MCP Server获取工具(tool)信息
2. 将tool schema作为输入，让LLM推断正确的执行顺序
3. 使用workflow.json作为参考答案进行评测
4. 多维度评测（顺序、依赖、数据流、控制流、拓扑）
"""

import json
import random
import copy
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import sys
import re
import asyncio
from tqdm import tqdm
from dataclasses import dataclass

# 假设这些模块存在
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from utils.llm_interface import LLMInterface
from core.config import LLMConfig
from utils.evaluate_tools import obfuscate_steps, parse_llm_output


# ==================== MCP Server Tools 获取模块 ====================

@dataclass
class MCPToolInfo:
    """MCP工具信息"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


class MCPServerToolsExtractor:
    """
    通过MCP协议从server.py提取工具信息
    支持两种方式：
    1. 使用fastmcp Client连接运行中的server (推荐)
    2. 直接导入server模块获取工具定义
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.server_path = project_path / "mcp_server" / "server.py"
        self.tools: List[MCPToolInfo] = []
    
    async def extract_tools_via_mcp_client(self, timeout: float = 30.0) -> List[MCPToolInfo]:
        """
        方法1：使用 uv run 启动 MCP Server 获取工具信息
        这会自动处理 pyproject.toml 中的依赖，实现环境隔离
        """
        try:
            # 使用 mcp 标准库，它提供了对启动命令的完全控制
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            import os
            
            # 设置环境变量
            env = os.environ.copy()
            # 强制 uv 使用非交互模式，并确保输出不包含进度条干扰 MCP 协议
            env["UV_NO_PROGRESS"] = "1" 
            env["PYTHONUNBUFFERED"] = "1"
            
            # 配置启动参数：使用 uv run 执行 server.py
            # 注意：cwd 设置为 project_path，这样 uv 才能找到 pyproject.toml
            server_params = StdioServerParameters(
                command="uv",
                args=["run", str(self.server_path)],
                cwd=str(self.project_path),
                env=env
            )
            
            extracted_tools = []
            
            # 建立连接
            # stdio_client 会启动 'uv run ...' 进程并管理通信
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 初始化 MCP 协议
                    await session.initialize()
                    
                    # 获取工具列表
                    tools_response = await session.list_tools()
                    
                    for tool in tools_response.tools:
                        tool_info = MCPToolInfo(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=tool.inputSchema or {}
                        )
                        extracted_tools.append(tool_info)
            
            self.tools = extracted_tools
            return extracted_tools
            
        except FileNotFoundError:
             print("⚠️ 'uv' command not found. Please ensure uv is installed and in your PATH.")
             return []
        except Exception as e:
            print(f"⚠️ MCP Client (uv run) connection failed: {e}")
            # 检查是否是 uv 执行错误
            if "exit code" in str(e):
                print(f"   💡 Tip: Check if {self.project_path}/pyproject.toml exists and is valid.")
                print(f"   💡 Tip: Try running 'uv run {self.server_path}' manually in that directory to debug.")
            return []

    # 保留原有的 import 方法作为备用
    async def extract_tools_via_import(self) -> List[MCPToolInfo]:
        """
        方法2：直接导入server模块获取工具定义
        """
        import importlib.util
        
        try:
            # 动态导入server模块
            spec = importlib.util.spec_from_file_location("server", self.server_path)
            server_module = importlib.util.module_from_spec(spec)
            
            # 添加项目路径到sys.path
            original_path = sys.path.copy()
            sys.path.insert(0, str(self.project_path))
            sys.path.insert(0, str(self.project_path / "mcp_server"))
            
            try:
                spec.loader.exec_module(server_module)
            except ImportError as e:
                print(f"   ❌ Import failed due to missing dependency: {e}")
                print(f"   👉 Please run: uv add {e.name}")
                raise e
            finally:
                sys.path = original_path
            
            tools = []
            
            # 查找fastmcp的mcp实例
            mcp_instance = None
            for name, obj in vars(server_module).items():
                # fastmcp 实例通常包含 _tools 属性
                if hasattr(obj, '_tools'): 
                    mcp_instance = obj
                    break
            
            if mcp_instance:
                # 从fastmcp实例获取工具
                if hasattr(mcp_instance, '_tools'):
                    for tool_name, tool_func in mcp_instance._tools.items():
                        description = tool_func.__doc__ or ""
                        input_schema = self._extract_function_schema(tool_func)
                        tools.append(MCPToolInfo(
                            name=tool_name,
                            description=description.strip(),
                            input_schema=input_schema
                        ))
            
            self.tools = tools
            return tools
            
        except Exception as e:
            print(f"❌ Failed to import server module: {e}")
            return []

    def _extract_function_schema(self, func) -> Dict:
        """根据函数签名生成输入参数的JSON Schema"""
        import inspect
        schema = {"type": "object", "properties": {}, "required": []}
        try:
            sig = inspect.signature(func)
            hints = func.__annotations__ if hasattr(func, '__annotations__') else {}
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'cls', 'ctx'): # fastmcp 可能会注入 ctx
                    continue
                prop = {"type": "string"}
                if param_name in hints:
                    hint = hints[param_name]
                    if hint == int: prop["type"] = "integer"
                    elif hint == float: prop["type"] = "number"
                    elif hint == bool: prop["type"] = "boolean"
                    elif hint == list: prop["type"] = "array"
                    elif hint == dict: prop["type"] = "object"
                schema["properties"][param_name] = prop
                if param.default == inspect.Parameter.empty:
                    schema["required"].append(param_name)
        except Exception:
            pass
        return schema

    async def extract_tools(self, method: str = "auto") -> List[MCPToolInfo]:
        """
        提取工具信息的统一入口
        """
        if not self.server_path.exists():
            print(f"❌ Server file not found: {self.server_path}")
            return []
        
        # 优先使用 fastmcp client，其次尝试 import
        methods = {
            "mcp_client": self.extract_tools_via_mcp_client,
            "import": self.extract_tools_via_import
        }
        
        if method == "auto":
            for method_name, extractor in methods.items():
                print(f"   Trying {method_name} method...")
                tools = await extractor()
                if tools:
                    print(f"   ✅ Successfully extracted {len(tools)} tools via {method_name}")
                    return tools
            print("   ❌ All extraction methods failed")
            return []
        else:
            if method in methods:
                return await methods[method]()
            else:
                raise ValueError(f"Unknown extraction method: {method}")

# ==================== 主评测系统 ====================

class WorkflowEvaluationSystem:
    """统一的工作流评测系统（MCP版本）"""
    
    def __init__(self, reference_workflow_path: str, seed: int = 42, obfuscate: bool = True):
        self.reference_workflow = self._load_workflow(reference_workflow_path)
        self.seed = seed
        self.obfuscate = obfuscate
        random.seed(seed)
        
        # 提取参考答案
        self.reference_answer = self._extract_reference_answer()
        self.id_mapping = None  # 存储混淆后的ID映射
        self.generated_workflow_steps = None  # 存储生成的工作流步骤
        self.mcp_tools: List[MCPToolInfo] = []  # 存储从server获取的工具信息
    
    def _load_workflow(self, path: str) -> Dict:
        """加载参考工作流"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_reference_answer(self) -> Dict:
        """从参考工作流中提取评测所需的所有信息"""
        steps = self.reference_workflow['workflow_steps']
        
        return {
            'execution_order': [s['step_id'] for s in steps],
            'steps_detail': {s['step_id']: s for s in steps},
            'dependencies': self._extract_dependencies(steps),
            'dataflow': self._extract_dataflow(steps),
            'control_flow': self._extract_control_flow(steps),
            'step_count': len(steps),
            'has_branches': self._check_has_branches(steps)
        }
    
    def _check_has_branches(self, steps: List[Dict]) -> bool:
        """检查工作流是否有分支结构"""
        for step in steps:
            next_steps = step.get('next_steps', [])
            if len(next_steps) > 1:
                return True
        return False
    
    def _extract_dependencies(self, steps: List[Dict]) -> Dict[str, List[str]]:
        """提取步骤间的依赖关系（next_steps）"""
        deps = {}
        for step in steps:
            if step.get('next_steps'):
                deps[step['step_id']] = step['next_steps']
        return deps
    
    def _extract_dataflow(self, steps: List[Dict]) -> Dict[str, List[str]]:
        """提取数据流依赖（{{step_X.output}}引用）"""
        dataflow = {}
        for step in steps:
            refs = self._find_step_references(step.get('parameters', {}))
            refs.extend(self._find_step_references(step.get('condition', '')))
            if refs:
                dataflow[step['step_id']] = list(set(refs))
        return dataflow
    
    def _extract_control_flow(self, steps: List[Dict]) -> Dict[str, Dict]:
        """提取控制流信息（else_steps, error_handler）"""
        control_flow = {}
        for step in steps:
            cf = {}
            if step.get('else_steps'):
                cf['else_steps'] = step['else_steps']
            if step.get('error_handler'):
                cf['error_handler'] = step['error_handler']
            if cf:
                control_flow[step['step_id']] = cf
        return control_flow
    
    def _find_step_references(self, obj: Any) -> List[str]:
        """递归查找所有{{step_<id>.*}}引用"""
        refs = []
        if isinstance(obj, dict):
            for v in obj.values():
                refs.extend(self._find_step_references(v))
        elif isinstance(obj, list):
            for x in obj:
                refs.extend(self._find_step_references(x))
        elif isinstance(obj, str):
            refs.extend(re.findall(r'\{\{\s*step_(\w+)[^}]*\}\}', obj))
        return refs
    
    def set_mcp_tools(self, tools: List[MCPToolInfo]):
        """设置从MCP Server获取的工具信息"""
        self.mcp_tools = tools
    
    def generate_shuffled_workflow(self) -> Dict:
        """
        生成打乱顺序的测试工作流
        
        步骤：
        1. 深拷贝原始工作流
        2. （可选）混淆步骤ID和文本
        3. 打乱步骤顺序
        4. 删除 next_steps 字段（让LLM自己推断）
        """
        shuffled = copy.deepcopy(self.reference_workflow)
        steps = shuffled['workflow_steps']
        
        # 记录原始ID顺序
        original_ids = [s['step_id'] for s in steps]
        
        # 1. 混淆步骤ID和文本（如果启用）
        if self.obfuscate:
            self.id_mapping = obfuscate_steps(steps)
            # 更新参考答案中的ID
            self.reference_answer['execution_order'] = [
                self.id_mapping[old_id] for old_id in original_ids
            ]
            # 更新依赖关系中的ID
            self.reference_answer['dependencies'] = {
                self.id_mapping[k]: [self.id_mapping[v] for v in vs]
                for k, vs in self.reference_answer['dependencies'].items()
            }
            # 更新数据流中的ID
            self.reference_answer['dataflow'] = {
                self.id_mapping[k]: [self.id_mapping[v] for v in vs]
                for k, vs in self.reference_answer['dataflow'].items()
            }
            # 更新控制流中的ID
            new_control_flow = {}
            for k, v in self.reference_answer['control_flow'].items():
                new_k = self.id_mapping[k]
                new_v = {}
                if 'else_steps' in v:
                    new_v['else_steps'] = [self.id_mapping[x] for x in v['else_steps']]
                if 'error_handler' in v:
                    new_v['error_handler'] = self.id_mapping[v['error_handler']]
                new_control_flow[new_k] = new_v
            self.reference_answer['control_flow'] = new_control_flow
        
        # 2. 打乱步骤顺序
        random.shuffle(steps)
        
        # 3. 删除 next_steps 字段（保留在参考答案中用于评测）
        for step in steps:
            step.pop('next_steps', None)
        
        shuffled['workflow_steps'] = steps
        return shuffled
    
    def build_prompt_with_mcp_tools(self, shuffled_workflow: Dict) -> str:
        """
        构建包含MCP Tools信息的prompt
        
        核心思路：
        1. 提供从server获取的tool schema信息
        2. 提供打乱顺序的workflow steps
        3. 让LLM根据tool的功能语义推断正确顺序
        """
        # 构建tools信息字符串
        tools_info = []
        for tool in self.mcp_tools:
            tool_dict = {
                "name": tool.name,
                "description": tool.description,
            }
            if tool.input_schema:
                tool_dict["parameters"] = tool.input_schema
            tools_info.append(tool_dict)
        
        tools_json = json.dumps(tools_info, indent=2, ensure_ascii=False)
        workflow_json = json.dumps(shuffled_workflow['workflow_steps'], indent=2, ensure_ascii=False)
        
        prompt = f"""You are an AI agent tasked with reconstructing a workflow whose steps have been shuffled.

**AVAILABLE MCP TOOLS:**
The following tools are available in this MCP server. Use their descriptions and parameter schemas to understand the semantic relationships between workflow steps:

{tools_json}

**IMPORTANT INSTRUCTIONS:**
- Step IDs are randomized (e.g., '3f94c550') and DO NOT reflect execution order
- DO NOT sort by step_id or by name alphabetically
- Use the tool descriptions above to understand what each step does
- Infer the correct execution order based on:
  1. Tool semantics: Match each step's "tool" field to the available tools above
  2. Logical workflow: validation → data retrieval → processing → conditional checks → output/termination
  3. Data dependencies: {{{{step_X.output}}}} means step X must execute before the current step
  4. Parameter references: Check if parameters reference outputs from other steps
  5. Conditional logic: steps may reference {{{{step_Y.condition_result}}}} to determine branching

**BRANCHING RULES:**
- If a conditional step has multiple outcome paths, include ALL target steps in its next_steps array
- Example: A conditional check may lead to both a success termination and a failure termination
- Both branches should be listed in the conditional step's next_steps: ["success_step_id", "failure_step_id"]

**OUTPUT FORMAT:**
Return ONLY a JSON object (no prose, no backticks, no markdown):
{{"workflow_steps": [{{"step_id": "<id>", "next_steps": ["<id>", ...]}}, ...]}}

**SHUFFLED WORKFLOW:**
{workflow_json}

Analyze each step's tool usage and parameters, then return the workflow_steps array in the correct execution order with proper next_steps relationships."""

        return prompt
    
    def build_prompt_without_mcp_tools(self, shuffled_workflow: Dict) -> str:
        """
        不使用MCP Tools信息的prompt（回退方案）
        保持与原版相同的逻辑
        """
        workflow_json = json.dumps(shuffled_workflow['workflow_steps'], indent=2, ensure_ascii=False)
        
        prompt = f"""You are an AI agent tasked with reconstructing a workflow whose steps have been shuffled.
**IMPORTANT INSTRUCTIONS:**
- Step IDs are randomized (e.g., '3f94c550') and DO NOT reflect execution order
- DO NOT sort by step_id or by name alphabetically
- Infer the correct execution order based on:
  1. Step types and tool semantics (validation → information → trigger → monitoring → conditional → termination → output)
  2. Data dependencies: {{{{step_X.output}}}} means step X must execute before the current step
  3. Conditional logic: steps may reference {{{{step_Y.condition_result}}}} to determine branching

**BRANCHING RULES:**
- If a conditional step has multiple outcome paths, include ALL target steps in its next_steps array
- Example: A conditional check may lead to both a success termination and a failure termination
- Both branches should be listed in the conditional step's next_steps: ["success_step_id", "failure_step_id"]

**OUTPUT FORMAT:**
Return ONLY a JSON object (no prose, no backticks, no markdown):
{{"workflow_steps": [{{"step_id": "<id>", "next_steps": ["<id>", ...]}}, ...]}}

**SHUFFLED WORKFLOW:**
{workflow_json}

Return the workflow_steps array in the correct execution order."""

        return prompt
    
    def evaluate(self, generated_workflow: Dict) -> Dict:
        """多维度评测生成的工作流"""
        gen_steps = generated_workflow['workflow_steps']
        gen_order = [s['step_id'] for s in gen_steps]
        
        self.generated_workflow_steps = gen_steps
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'reference_order': self.reference_answer['execution_order'],
            'generated_order': gen_order,
            'metrics': {}
        }
        
        # 1. 执行顺序准确性
        results['metrics']['order_accuracy'] = self._evaluate_order(gen_order)
        
        # 2. 依赖关系准确性
        results['metrics']['dependency_accuracy'] = self._evaluate_dependencies(gen_steps)
        
        # 3. 控制流准确性
        results['metrics']['control_flow_accuracy'] = self._evaluate_control_flow(gen_steps)
        
        # 计算总分
        results['overall_score'] = self._calculate_overall_score(results['metrics'])
        
        return results
    
    def _evaluate_order(self, generated_order: List[str]) -> Dict:
        """评测执行顺序准确性"""
        ref_order = self.reference_answer['execution_order']
        ref_deps = self.reference_answer['dependencies']
        
        has_branches = any(len(nexts) > 1 for nexts in ref_deps.values())
        
        if has_branches:
            return self._evaluate_branching_workflow(self.generated_workflow_steps)
        else:
            return self._evaluate_linear_order(generated_order, ref_order)
    
    def _evaluate_branching_workflow(self, gen_steps: List[Dict]) -> Dict:
        """针对分支工作流的评测"""
        ref_deps = self.reference_answer['dependencies']
        ref_order = self.reference_answer['execution_order']
        
        gen_deps = {}
        for step in gen_steps:
            if step.get('next_steps'):
                gen_deps[step['step_id']] = step['next_steps']
        
        gen_order = [s['step_id'] for s in gen_steps]
        
        results = {'evaluation_type': 'branching_workflow'}
        
        ref_branch_points = [(k, v) for k, v in ref_deps.items() if len(v) > 1]
        
        if not ref_branch_points:
            return self._evaluate_linear_order(gen_order, ref_order)
        
        dependency_result = self._evaluate_dependency_satisfaction(ref_deps, gen_order)
        
        ref_steps_set = set(ref_order)
        gen_steps_set = set(gen_order)
        missing_steps = ref_steps_set - gen_steps_set
        extra_steps = gen_steps_set - ref_steps_set
        
        completeness = len(ref_steps_set & gen_steps_set) / len(ref_steps_set) if ref_steps_set else 1.0
        
        overall_path_correctness = dependency_result['satisfaction_rate'] * completeness
        
        results['overall_path_correctness'] = overall_path_correctness
        results['overall_path_details'] = {
            'dependency_satisfaction': dependency_result['satisfaction_rate'],
            'completeness': completeness,
            'missing_steps': list(missing_steps),
            'extra_steps': list(extra_steps),
            'satisfied_dependencies': dependency_result['satisfied_dependencies'],
            'total_dependencies': dependency_result['total_dependencies'],
            'violations': dependency_result['violations']
        }
        
        return results
    
    def _extract_linear_path(self, start_step: str, dependencies: Dict[str, List[str]]) -> List[str]:
        """从某个步骤开始，提取到终点的完整线性路径"""
        path = [start_step]
        current = start_step
        visited = {start_step}
        
        while current in dependencies and dependencies[current]:
            next_steps = dependencies[current]
            
            if not next_steps:
                break
            
            next_step = next_steps[0]
            
            if next_step in visited:
                break
            
            path.append(next_step)
            visited.add(next_step)
            current = next_step
        
        return path
    
    def _evaluate_dependency_satisfaction(self, ref_deps: Dict[str, List[str]], gen_order: List[str]) -> Dict:
        """评测依赖关系的满足度"""
        gen_positions = {step_id: i for i, step_id in enumerate(gen_order)}
        
        violations = []
        satisfied_deps = 0
        total_deps = 0
        
        for step_id, next_steps in ref_deps.items():
            for next_id in next_steps:
                total_deps += 1
                
                if step_id not in gen_positions or next_id not in gen_positions:
                    violations.append({
                        'from': step_id,
                        'to': next_id,
                        'reason': f'Missing step: {step_id if step_id not in gen_positions else next_id}'
                    })
                    continue
                
                if gen_positions[step_id] < gen_positions[next_id]:
                    satisfied_deps += 1
                else:
                    violations.append({
                        'from': step_id,
                        'to': next_id,
                        'from_pos': gen_positions[step_id],
                        'to_pos': gen_positions[next_id],
                        'reason': f'Dependency violated: {step_id} must come before {next_id}'
                    })
        
        satisfaction_rate = satisfied_deps / total_deps if total_deps > 0 else 1.0
        
        return {
            'satisfaction_rate': satisfaction_rate,
            'satisfied_dependencies': satisfied_deps,
            'total_dependencies': total_deps,
            'violations': violations,
            'is_valid': len(violations) == 0
        }
    
    def _evaluate_linear_order(self, generated_order: List[str], ref_order: List[str]) -> Dict:
        """评测线性顺序"""
        exact_match = (generated_order == ref_order)
        lcs_score = self._calculate_lcs_similarity(generated_order, ref_order)
        position_errors = self._calculate_position_errors(generated_order, ref_order)
        
        ref_steps_set = set(ref_order)
        gen_steps_set = set(generated_order)
        completeness = len(ref_steps_set & gen_steps_set) / len(ref_steps_set) if ref_steps_set else 1.0
        
        return {
            'evaluation_type': 'linear',
            'exact_match': exact_match,
            'overall_path_correctness': lcs_score * completeness,
            'lcs_similarity': lcs_score,
            'completeness': completeness,
            'position_errors': position_errors
        }
    
    def _calculate_lcs_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """计算两个序列的LCS相似度"""
        if not seq1 or not seq2:
            return 0.0
        
        if len(seq1) < len(seq2):
            seq1, seq2 = seq2, seq1
        
        m, n = len(seq1), len(seq2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev
        
        lcs_length = prev[n]
        return lcs_length / max(m, n)
    
    def _calculate_position_errors(self, gen_order: List[str], ref_order: List[str]) -> Dict:
        """计算位置偏差"""
        ref_positions = {sid: i for i, sid in enumerate(ref_order)}
        
        errors = []
        for i, sid in enumerate(gen_order):
            if sid in ref_positions:
                error = abs(i - ref_positions[sid])
                errors.append(error)
        
        if not errors:
            return {'mean': 0, 'max': 0, 'total': 0}
        
        return {
            'mean': sum(errors) / len(errors),
            'max': max(errors),
            'total': sum(errors)
        }
    
    def _evaluate_dependencies(self, gen_steps: List[Dict]) -> Dict:
        """评测依赖关系准确性（next_steps）"""
        ref_deps = self.reference_answer['dependencies']
        
        gen_deps = {}
        for step in gen_steps:
            if step.get('next_steps'):
                gen_deps[step['step_id']] = set(step['next_steps'])
        
        total_edges = 0
        correct_edges = 0
        missing_edges = []
        extra_edges = []
        
        for sid, ref_nexts in ref_deps.items():
            for next_id in ref_nexts:
                total_edges += 1
                if next_id in gen_deps.get(sid, set()):
                    correct_edges += 1
                else:
                    missing_edges.append((sid, next_id))
        
        for sid, gen_nexts in gen_deps.items():
            for next_id in gen_nexts:
                if sid not in ref_deps or next_id not in ref_deps[sid]:
                    extra_edges.append((sid, next_id))
        
        accuracy = correct_edges / total_edges if total_edges > 0 else 1.0
        
        return {
            'accuracy': accuracy,
            'total_edges': total_edges,
            'correct_edges': correct_edges,
            'missing_edges': missing_edges,
            'extra_edges': extra_edges,
            'score': accuracy
        }
    
    def _evaluate_control_flow(self, gen_steps: List[Dict]) -> Dict:
        """评测控制流准确性（else_steps, error_handler）"""
        ref_control = self.reference_answer['control_flow']
        
        if not ref_control:
            return {'accuracy': 1.0, 'score': 1.0, 'details': 'No control flow in reference'}
        
        gen_control = {}
        for step in gen_steps:
            cf = {}
            if step.get('else_steps'):
                cf['else_steps'] = set(step['else_steps'])
            if step.get('error_handler'):
                cf['error_handler'] = step['error_handler']
            if cf:
                gen_control[step['step_id']] = cf
        
        total_controls = 0
        correct_controls = 0
        details = {'else_steps': {}, 'error_handler': {}}
        
        for sid, ref_cf in ref_control.items():
            gen_cf = gen_control.get(sid, {})
            
            if 'else_steps' in ref_cf:
                total_controls += len(ref_cf['else_steps'])
                ref_else = set(ref_cf['else_steps'])
                gen_else = gen_cf.get('else_steps', set())
                correct = len(ref_else & gen_else)
                correct_controls += correct
                details['else_steps'][sid] = {
                    'expected': list(ref_else),
                    'actual': list(gen_else),
                    'correct': correct
                }
            
            if 'error_handler' in ref_cf:
                total_controls += 1
                if gen_cf.get('error_handler') == ref_cf['error_handler']:
                    correct_controls += 1
                    details['error_handler'][sid] = 'correct'
                else:
                    details['error_handler'][sid] = {
                        'expected': ref_cf['error_handler'],
                        'actual': gen_cf.get('error_handler', None)
                    }

        accuracy = correct_controls / total_controls if total_controls > 0 else 1.0
        return {
            'accuracy': accuracy,
            'total_controls': total_controls,
            'correct_controls': correct_controls,
            'details': details,
            'score': accuracy
        }
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """计算总分"""
        order_metrics = metrics.get('order_accuracy', {})
        order_score = order_metrics.get('overall_path_correctness', 0.0)
        
        weights = {
            'order': 0.50,
            'dependency_accuracy': 0.25,
            'control_flow_accuracy': 0.25,
        }
        
        total_score = (
            weights['order'] * order_score +
            weights['dependency_accuracy'] * metrics.get('dependency_accuracy', {}).get('score', 0.0) +
            weights['control_flow_accuracy'] * metrics.get('control_flow_accuracy', {}).get('score', 0.0)
        )
        
        return total_score
    
    def generate_report(self, results: Dict, output_path: Path = None) -> str:
        """生成详细评测报告"""
        lines = [
            "=" * 80,
            "Workflow Orchestration Evaluation Report (MCP Version)",
            "=" * 80,
            f"Timestamp: {results['timestamp']}",
            f"MCP Tools Used: {len(self.mcp_tools)} tools",
            "",
        ]
        
        # 显示使用的MCP工具
        if self.mcp_tools:
            lines.append("📦 MCP Tools:")
            for tool in self.mcp_tools:
                lines.append(f"   - {tool.name}: {tool.description[:60]}...")
            lines.append("")
        
        # 执行顺序评测结果
        order_metrics = results['metrics']['order_accuracy']
        
        if order_metrics.get('evaluation_type') == 'branching_workflow':
            lines.extend([
                "=" * 80,
                "EXECUTION ORDER EVALUATION (BRANCHING WORKFLOW)",
                "=" * 80,
                "",
                "📊 KEY METRICS:",
                "",
                f"1. Overall Path Correctness:      {order_metrics['overall_path_correctness']:.2%}"
            ])
            
            lines.extend([
                "=" * 80,
                "DETAILED ANALYSIS",
                "=" * 80,
                "",
            ])
            
            overall_details = order_metrics['overall_path_details']
            lines.extend([
                "1️⃣  OVERALL PATH CORRECTNESS",
                f"   - Dependency Satisfaction: {overall_details['dependency_satisfaction']:.2%}",
                f"   - Completeness: {overall_details['completeness']:.2%}",
                f"   - Satisfied Dependencies: {overall_details['satisfied_dependencies']}/{overall_details['total_dependencies']}",
            ])
            
            if overall_details['missing_steps']:
                lines.append(f"   - Missing Steps: {overall_details['missing_steps']}")
            if overall_details['extra_steps']:
                lines.append(f"   - Extra Steps: {overall_details['extra_steps']}")
            
            if overall_details['violations']:
                lines.append(f"   - Violations: {len(overall_details['violations'])}")
                for v in overall_details['violations'][:3]:
                    lines.append(f"     • {v['reason']}")
            
            lines.append("")
            
        else:
            lines.extend([
                "=" * 80,
                "EXECUTION ORDER EVALUATION (LINEAR WORKFLOW)",
                "=" * 80,
                "",
                "📊 KEY METRIC:",
                "",
                f"Overall Path Correctness: {order_metrics['overall_path_correctness']:.2%}",
                "",
                "=" * 80,
                "DETAILED ANALYSIS",
                "=" * 80,
                "",
                f"- Exact Match: {'✓' if order_metrics['exact_match'] else '✗'}",
                f"- LCS Similarity: {order_metrics.get('lcs_similarity', 0):.2%}",
                f"- Completeness: {order_metrics.get('completeness', 0):.2%}",
                f"- Mean Position Error: {order_metrics['position_errors']['mean']:.2f}",
                f"- Max Position Error: {order_metrics['position_errors']['max']}",
                "",
            ])
        
        # 其他指标
        lines.extend([
            "=" * 80,
            "OTHER METRICS",
            "=" * 80,
            "",
        ])
        
        dep_metrics = results['metrics']['dependency_accuracy']
        lines.extend([
            "Dependency Accuracy (next_steps)",
            f"- Accuracy: {dep_metrics['accuracy']:.2%}",
            f"- Correct Edges: {dep_metrics['correct_edges']}/{dep_metrics['total_edges']}",
            f"- Missing Edges: {len(dep_metrics['missing_edges'])}",
            f"- Extra Edges: {len(dep_metrics['extra_edges'])}",
            "",
        ])
        
        cf_metrics = results['metrics']['control_flow_accuracy']
        lines.extend([
            "Control Flow Accuracy",
            f"- Accuracy: {cf_metrics['accuracy']:.2%}",
            "",
        ])
        
        # 可视化对比
        lines.extend([
            "=" * 80,
            "WORKFLOW STRUCTURE COMPARISON",
            "=" * 80,
            "",
        ])
        
        gen_deps = {}
        if hasattr(self, 'generated_workflow_steps'):
            for step in self.generated_workflow_steps:
                if step.get('next_steps'):
                    gen_deps[step['step_id']] = step['next_steps']
        
        if self.reference_answer.get('has_branches'):
            lines.append("📋 Reference Structure:")
            ref_deps = self.reference_answer['dependencies']
            
            branch_points = [(k, v) for k, v in ref_deps.items() if len(v) > 1]
            if branch_points:
                branch_point_id, branch_targets = branch_points[0]
                branch_index = self.reference_answer['execution_order'].index(branch_point_id)
                main_path = self.reference_answer['execution_order'][:branch_index + 1]
                
                lines.append(f"  Main Path: {' -> '.join(main_path)}")
                lines.append(f"  Branches from {branch_point_id}:")
                
                for i, branch_id in enumerate(branch_targets, 1):
                    branch_path = self._extract_linear_path(branch_id, ref_deps)
                    lines.append(f"    Branch {i}: {' -> '.join(branch_path)}")
        else:
            lines.extend([
                "📋 Reference Order:",
                f"  {' -> '.join(results['reference_order'])}",
            ])
        
        lines.append("")
        
        has_gen_branches = any(len(nexts) > 1 for nexts in gen_deps.values())
        
        if has_gen_branches:
            lines.append("🤖 Generated Structure:")
            
            gen_branch_points = [(k, v) for k, v in gen_deps.items() if len(v) > 1]
            if gen_branch_points:
                gen_branch_point_id, gen_branch_targets = gen_branch_points[0]
                
                gen_order = results['generated_order']
                branch_index = gen_order.index(gen_branch_point_id)
                gen_main_path = gen_order[:branch_index + 1]
                
                lines.append(f"  Main Path: {' -> '.join(gen_main_path)}")
                lines.append(f"  Branches from {gen_branch_point_id}:")
                
                for i, branch_id in enumerate(gen_branch_targets, 1):
                    branch_path = self._extract_linear_path(branch_id, gen_deps)
                    lines.append(f"    Branch {i}: {' -> '.join(branch_path)}")
        else:
            lines.extend([
                "🤖 Generated Order:",
                f"  {' -> '.join(results['generated_order'])}",
            ])
        
        lines.extend([
            "",
            "=" * 80,
            "OVERALL SCORE",
            "=" * 80,
            f"Total Score: {results['overall_score']:.2%}",
            "",
        ])
        
        if results['overall_score'] >= 0.95:
            lines.append(f"✅ EXCELLENT")
        elif results['overall_score'] >= 0.80:
            lines.append(f"✅ PASSED")
        elif results['overall_score'] >= 0.60:
            lines.append(f"⚠️  NEEDS IMPROVEMENT")
        else:
            lines.append(f"❌ FAILED")
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


# ==================== 主函数 ====================

async def main():
    print("=" * 80)
    print("Unified Workflow Orchestration Evaluation System (MCP Version)")
    print("=" * 80)

    # 配置LLM
    llm_config = LLMConfig(
        provider="azure",
        api_key="282edc7433594a788ce28f3b0572dd2a",  # 替换为实际的API Key
        base_url="https://gpt.yunstorm.com/",
        api_version="2025-04-01-preview",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=2048,
        timeout=60,
        retry_attempts=3,
        retry_delay=2.0
    )
    llm_interface = LLMInterface(llm_config)

    # 根目录路径
    reference_workflow_root_path = Path('E:\\MCPBenchMark\\MCPFLow\\mcp_projects')
    
    # 输出根目录
    output_root_dir = Path('./evaluation_output_mcp')
    output_root_dir.mkdir(exist_ok=True)
    
    # 收集所有 mcp_project 目录
    mcp_projects = []
    for item in reference_workflow_root_path.iterdir():
        if item.is_dir() and item.name.startswith('mcp_project'):
            try:
                project_num = int(item.name.replace('mcp_project', ''))
                mcp_projects.append((project_num, item))
            except ValueError:
                print(f"⚠️ Skipping invalid project name: {item.name}")
                continue
    
    mcp_projects.sort(key=lambda x: x[0])
    
    print(f"\n📁 Found {len(mcp_projects)} mcp_project directories")
    
    # 汇总结果
    summary_results = {
        'total': len(mcp_projects),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'details': []
    }
    
    # 遍历每个项目
    for project_num, project_path in tqdm(mcp_projects, total=len(mcp_projects), desc="Processing Projects"):

        if project_num != 23: 
            continue 

        print(f"\n{'='*60}")
        print(f"Processing mcp_project{project_num}")
        print(f"{'='*60}")
        
        # 创建该项目的输出目录
        project_output_dir = output_root_dir / str(project_num)
        project_output_dir.mkdir(exist_ok=True)
        
        workflow_path = project_path / 'workflow.json'
        server_path = project_path / 'mcp_server' / 'server.py'
        
        # 检查必要文件是否存在
        if not workflow_path.exists():
            skip_msg = f"workflow.json not found in mcp_project{project_num}"
            print(f"⚠️ {skip_msg}")
            
            with open(project_output_dir / 'skipped.txt', 'w', encoding='utf-8') as f:
                f.write(f"Skipped: {skip_msg}\n")
                f.write(f"Project path: {project_path}\n")
            
            summary_results['skipped'] += 1
            summary_results['details'].append({
                'project_num': project_num,
                'status': 'skipped',
                'reason': 'workflow.json not found'
            })
            continue
        
        if not server_path.exists():
            skip_msg = f"server.py not found in mcp_project{project_num}"
            print(f"⚠️ {skip_msg}")
            
            with open(project_output_dir / 'skipped.txt', 'w', encoding='utf-8') as f:
                f.write(f"Skipped: {skip_msg}\n")
                f.write(f"Project path: {project_path}\n")
            
            summary_results['skipped'] += 1
            summary_results['details'].append({
                'project_num': project_num,
                'status': 'skipped',
                'reason': 'server.py not found'
            })
            continue
        
        try:
            # ===== 1. 从MCP Server获取工具信息 =====
            print(f"\n🔧 Extracting MCP tools from server.py...")
            extractor = MCPServerToolsExtractor(project_path)
            mcp_tools = await extractor.extract_tools(method="auto")
            
            if not mcp_tools:
                print(f"⚠️ No tools extracted, falling back to workflow.json only")
            else:
                print(f"✅ Extracted {len(mcp_tools)} tools:")
                for tool in mcp_tools:
                    print(f"   - {tool.name}")
            
            # 保存工具信息
            tools_info = [t.to_dict() for t in mcp_tools]
            with open(project_output_dir / 'mcp_tools.json', 'w', encoding='utf-8') as f:
                json.dump(tools_info, f, indent=2, ensure_ascii=False)
            
            # ===== 2. 初始化评测系统 =====
            eval_system = WorkflowEvaluationSystem(
                reference_workflow_path=str(workflow_path),
                seed=42,
                obfuscate=True
            )
            eval_system.set_mcp_tools(mcp_tools)
            
            # ===== 3. 生成打乱的测试工作流 =====
            shuffled_workflow = eval_system.generate_shuffled_workflow()
            
            with open(project_output_dir / 'shuffled_workflow.json', 'w', encoding='utf-8') as f:
                json.dump(shuffled_workflow, f, indent=2, ensure_ascii=False)
            
            with open(project_output_dir / 'reference_answer.json', 'w', encoding='utf-8') as f:
                json.dump(eval_system.reference_answer, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Generated shuffled workflow (Obfuscated: {eval_system.obfuscate})")
            print(f"   Steps: {len(shuffled_workflow['workflow_steps'])}")
            
            if eval_system.reference_answer.get('has_branches'):
                print(f"   Workflow Type: BRANCHING (DAG)")
            else:
                print(f"   Workflow Type: LINEAR")
            
            # ===== 4. 构建Prompt（使用MCP Tools信息） =====
            if mcp_tools:
                prompt = eval_system.build_prompt_with_mcp_tools(shuffled_workflow)
                print(f"\n📤 Sending to LLM (with {len(mcp_tools)} MCP tools)...")
            else:
                prompt = eval_system.build_prompt_without_mcp_tools(shuffled_workflow)
                print(f"\n📤 Sending to LLM (without MCP tools)...")
            
            print(f"   Prompt length: {len(prompt)} characters")
            
            # 保存prompt
            with open(project_output_dir / 'prompt.txt', 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            # ===== 5. 调用LLM =====
            try:
                llm_output = await llm_interface.generate_response(prompt)
                with open(project_output_dir / 'llm_response.txt', 'w', encoding='utf-8') as f:
                    f.write(llm_output)
            except asyncio.TimeoutError:
                raise Exception("LLM request timed out")
            except Exception as e:
                raise Exception(f"LLM request failed: {type(e).__name__}: {e}")
            
            # 保存LLM原始输出
            with open(project_output_dir / 'llm_raw_output.txt', 'w', encoding='utf-8') as f:
                f.write(llm_output)
            
            # ===== 6. 解析LLM输出 =====
            try:
                generated_workflow = parse_llm_output(llm_output)
                print("✅ Successfully parsed LLM output")
                
                with open(project_output_dir / 'generated_workflow.json', 'w', encoding='utf-8') as f:
                    json.dump(generated_workflow, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                raise Exception(f"Failed to parse LLM output: {e}")
            
            # ===== 7. 评测 =====
            print("\n🔍 Evaluating...")
            results = eval_system.evaluate(generated_workflow)
            
            # ===== 8. 生成报告 =====
            report = eval_system.generate_report(
                results,
                output_path=project_output_dir / 'evaluation_report.txt'
            )
            
            print("\n" + report)
            
            # 保存详细结果
            with open(project_output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            summary_results['success'] += 1
            summary_results['details'].append({
                'project_num': project_num,
                'status': 'success',
                'mcp_tools_count': len(mcp_tools),
                'results': results
            })
            
            print(f"\n✅ mcp_project{project_num} evaluation complete!")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error processing mcp_project{project_num}: {error_msg}")
            
            with open(project_output_dir / 'error.txt', 'w', encoding='utf-8') as f:
                f.write(f"Error: {error_msg}\n")
                f.write(f"Project path: {project_path}\n")
            
            summary_results['failed'] += 1
            summary_results['details'].append({
                'project_num': project_num,
                'status': 'failed',
                'error': error_msg
            })
    
    # 生成汇总报告
    print(f"\n{'='*80}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total projects: {summary_results['total']}")
    print(f"Success: {summary_results['success']}")
    print(f"Skipped (no workflow.json/server.py): {summary_results['skipped']}")
    print(f"Failed: {summary_results['failed']}")
    
    # 保存汇总结果
    with open(output_root_dir / 'batch_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    # 生成可读的汇总报告
    summary_report = generate_summary_report(summary_results)
    with open(output_root_dir / 'batch_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    print(f"\n✅ Batch evaluation complete!")
    print(f"   Results saved to: {output_root_dir.absolute()}")


def generate_summary_report(summary_results):
    """生成可读的汇总报告"""
    lines = [
        "=" * 80,
        "BATCH EVALUATION SUMMARY REPORT (MCP Version)",
        "=" * 80,
        "",
        f"Total Projects: {summary_results['total']}",
        f"Successful: {summary_results['success']}",
        f"Skipped: {summary_results['skipped']}",
        f"Failed: {summary_results['failed']}",
        "",
        "-" * 80,
        "DETAILED RESULTS",
        "-" * 80,
        ""
    ]
    
    for detail in summary_results['details']:
        project_num = detail['project_num']
        status = detail['status']
        
        if status == 'success':
            results = detail.get('results', {})
            score = results.get('overall_score', 'N/A')
            tools_count = detail.get('mcp_tools_count', 0)
            lines.append(f"Project {project_num}: ✅ SUCCESS (Score: {score:.2%}, Tools: {tools_count})")
        elif status == 'skipped':
            reason = detail.get('reason', 'Unknown')
            lines.append(f"Project {project_num}: ⚠️ SKIPPED ({reason})")
        else:
            error = detail.get('error', 'Unknown error')
            lines.append(f"Project {project_num}: ❌ FAILED ({error[:50]}...)")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


if __name__ == '__main__':
    asyncio.run(main())
