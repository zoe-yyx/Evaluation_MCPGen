"""
tasks/mcp_tool_regenerator.py (同时返回两种测试结果)
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

from core.config import EvaluationConfig
from utils.llm_interface import LLMInterface
from tasks.comprehensive_metrics import ComprehensiveToolMetrics, ComprehensiveMetricsAnalyzer
from tasks.isolated_test_env import IsolatedTestEnvironment, TestRunner
from tasks.improved_prompts import ImprovedPromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class IndividualTestResult:
    """单工具测试结果"""
    tool_name: str
    success: bool
    execution_time: float = 0.0
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool_name': self.tool_name,
            'success': self.success,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
        }


@dataclass 
class CollectiveTestResult:
    """整体测试结果"""
    success: bool
    execution_time: float = 0.0
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    tools_tested: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'tools_tested': self.tools_tested,
        }


@dataclass
class RegeneratedTool:
    """重新生成的工具信息"""
    
    name: str
    code: str
    
    original_path: Optional[Path] = None
    generated_path: Optional[Path] = None
    
    # 静态分析指标
    static_metrics: Optional[ComprehensiveToolMetrics] = None
    
    # 单工具测试结果
    individual_test: Optional[IndividualTestResult] = None
    
    generation_time: float = 0.0
    prompt_used: Optional[str] = None
    
    @property
    def individual_test_success(self) -> bool:
        return self.individual_test.success if self.individual_test else False
    
    @property
    def static_analysis_passed(self) -> bool:
        if not self.static_metrics:
            return False
        return self.static_metrics.syntax_valid and self.static_metrics.has_target_function
    
    def to_dict(self, include_code: bool = False, include_prompt: bool = False) -> Dict[str, Any]:
        result = {
            'name': self.name,
            'original_path': str(self.original_path) if self.original_path else None,
            'generated_path': str(self.generated_path) if self.generated_path else None,
            'generation_time': self.generation_time,
            'static_analysis_passed': self.static_analysis_passed,
            'individual_test_success': self.individual_test_success,
        }
        
        if include_code:
            result['code'] = self.code
        if include_prompt:
            result['prompt_used'] = self.prompt_used
        if self.static_metrics:
            result['static_metrics'] = self.static_metrics.to_dict()
        if self.individual_test:
            result['individual_test'] = self.individual_test.to_dict()
        
        return result


@dataclass
class RegenerationResult:
    """重新生成的结果（包含两种测试结果）"""
    
    workflow_id: str
    workflow_name: str
    tools: List[RegeneratedTool] = field(default_factory=list)
    
    # === 工具统计 ===
    total_tools: int = 0
    static_analysis_passed: int = 0
    
    # === 单工具测试结果 ===
    individual_test_passed: int = 0  # 单工具测试通过的数量
    
    # === 整体测试结果 ===
    collective_test: Optional[CollectiveTestResult] = None
    
    # === 高质量工具 ===
    high_quality_tools: int = 0  # 综合得分 > 0.8
    
    # === 时间统计 ===
    total_time: float = 0.0
    average_generation_time: float = 0.0
    
    error_message: Optional[str] = None
    
    @property
    def collective_test_success(self) -> bool:
        return self.collective_test.success if self.collective_test else False
    
    @property
    def all_individual_tests_passed(self) -> bool:
        return self.individual_test_passed == self.total_tools and self.total_tools > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'workflow_name': self.workflow_name,
            
            # 工具统计
            'total_tools': self.total_tools,
            'static_analysis_passed': self.static_analysis_passed,
            'static_pass_rate': self.static_analysis_passed / self.total_tools if self.total_tools > 0 else 0.0,
            
            # 单工具测试结果
            'individual_test': {
                'passed': self.individual_test_passed,
                'pass_rate': self.individual_test_passed / self.total_tools if self.total_tools > 0 else 0.0,
                'all_passed': self.all_individual_tests_passed,
            },
            
            # 整体测试结果
            'collective_test': self.collective_test.to_dict() if self.collective_test else None,
            
            # 高质量统计
            'high_quality_tools': self.high_quality_tools,
            'high_quality_rate': self.high_quality_tools / self.total_tools if self.total_tools > 0 else 0.0,
            
            # 时间
            'total_time': self.total_time,
            'average_generation_time': self.average_generation_time,
            
            'error_message': self.error_message,
            'tools': [tool.to_dict() for tool in self.tools],
        }


class MCPToolRegenerator:
    """MCP 工具重新生成器"""
    
    def __init__(self, config: EvaluationConfig, llm_interface: LLMInterface):
        self.config = config
        self.llm = llm_interface
    
    async def regenerate_tool(
        self,
        workflow_dir: Path,
        tool_name: str,
        original_tool_path: Optional[Path] = None
    ) -> RegeneratedTool:
        """重新生成单个工具"""
        start_time = time.time()
        
        tool = RegeneratedTool(
            name=tool_name,
            code="",
            original_path=original_tool_path
        )
        
        try:
            prompt_builder = ImprovedPromptBuilder(workflow_dir)
            prompt = prompt_builder.build_tool_generation_prompt(
                tool_name=tool_name,
                original_tool_path=original_tool_path,
                include_examples=True
            )
            tool.prompt_used = prompt
            
            logger.info(f"Generating code for tool: {tool_name}")
            raw_code = await self._generate_code(prompt)
            
            tool.code = self._clean_generated_code(raw_code)
            tool.generation_time = time.time() - start_time
            logger.info(f"Generated tool '{tool_name}' in {tool.generation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to generate tool '{tool_name}': {e}")
            tool.generation_time = time.time() - start_time
        
        return tool
    
    async def _generate_code(self, prompt: str) -> str:
        """调用 LLM 生成代码"""
        system_message = (
            "You are an expert Python developer specializing in FastMCP tool development. "
            "Generate clean, production-quality code that follows all requirements exactly. "
            "Output ONLY the function code, no markdown, no explanations."
        )
        
        response = await self.llm.generate_response(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2,
            max_tokens=2000,
        )
        
        return response if response else ""
    
    def _clean_generated_code(self, raw_code: str) -> str:
        """清理生成的代码"""
        code = raw_code.strip()
        
        code = re.sub(r'^```(?:python)?\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'@mcp\.tool\(\)\s*\n', '', code)
        
        lines = code.split('\n')
        cleaned_lines = []
        prev_blank = False
        
        for line in lines:
            is_blank = line.strip() == ''
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line)
            prev_blank = is_blank
        
        return '\n'.join(cleaned_lines).strip()
    
    def _perform_static_analysis(self, tool: RegeneratedTool) -> ComprehensiveToolMetrics:
        """执行静态分析"""
        analyzer = ComprehensiveMetricsAnalyzer(
            original_tool_path=tool.original_path
        )
        print(f"Performing static analysis for tool: {tool.name}")
        print("tool.original_path", tool.original_path)
        
        metrics = analyzer.analyze(
            generated_code=tool.code,
            tool_name=tool.name,
            execution_result=None
        )
        return metrics
    
    async def _run_individual_test(
        self,
        workflow_dir: Path,
        tool: RegeneratedTool
    ) -> IndividualTestResult:
        """
        单工具测试：只替换这一个工具，其他保持原始版本
        
        评测目标：生成的工具是否与原工作流兼容
        """
        logger.info(f"[Individual Test] Testing tool: {tool.name}")
        
        result = IndividualTestResult(tool_name=tool.name, success=False)
        
        if not tool.code:
            result.error_message = "No code generated"
            return result
        
        with IsolatedTestEnvironment(workflow_dir, keep_on_error=False) as env:
            try:
                # 只替换这一个工具
                env.install_regenerated_tool(tool.name, tool.code)
                
                # 运行工作流
                exec_result = env.run_workflow_test(
                    timeout=self.config.e2e_config.execution_timeout
                )
                
                result.success = exec_result.get('success', False)
                result.execution_time = exec_result.get('execution_time', 0.0)
                result.error_message = exec_result.get('error')
                result.stdout = exec_result.get('stdout')
                result.stderr = exec_result.get('stderr')
                
            except Exception as e:
                logger.error(f"Individual test failed for '{tool.name}': {e}")
                result.error_message = str(e)
        
        status = "✅ PASS" if result.success else "❌ FAIL"
        logger.info(f"[Individual Test] {tool.name}: {status}")
        return result
    
    async def _run_collective_test(
        self,
        workflow_dir: Path,
        tools: List[RegeneratedTool]
    ) -> CollectiveTestResult:
        """
        整体测试：所有工具一起替换
        
        评测目标：所有生成的工具是否能协同工作
        """
        # 只测试静态分析通过的工具
        valid_tools = [t for t in tools if t.static_analysis_passed and t.code]
        
        logger.info(f"[Collective Test] Testing {len(valid_tools)}/{len(tools)} tools together")
        
        result = CollectiveTestResult(
            success=False,
            tools_tested=[t.name for t in valid_tools]
        )
        
        if not valid_tools:
            result.error_message = "No valid tools to test"
            return result
        
        with IsolatedTestEnvironment(workflow_dir, keep_on_error=False) as env:
            try:
                # 替换所有工具
                for tool in valid_tools:
                    env.install_regenerated_tool(tool.name, tool.code)
                    logger.debug(f"Installed tool: {tool.name}")
                
                # 运行工作流
                exec_result = env.run_workflow_test(
                    timeout=self.config.e2e_config.execution_timeout
                )
                
                result.success = exec_result.get('success', False)
                result.execution_time = exec_result.get('execution_time', 0.0)
                result.error_message = exec_result.get('error')
                result.stdout = exec_result.get('stdout')
                result.stderr = exec_result.get('stderr')
                
            except Exception as e:
                logger.error(f"Collective test failed: {e}")
                result.error_message = str(e)
        
        status = "✅ PASS" if result.success else "❌ FAIL"
        logger.info(f"[Collective Test] {status}")
        return result
    
    async def run_regeneration_evaluation(
        self,
        workflow_dir: Path,
        output_dir: Path
    ) -> RegenerationResult:
        """
        运行完整的重新生成和评测流程
        
        同时执行两种测试：
        1. 单工具测试（Individual）：每个工具单独替换
        2. 整体测试（Collective）：所有工具一起替换
        
        Returns:
            RegenerationResult: 包含两种测试结果
        """
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Starting regeneration evaluation for workflow: {workflow_dir.name} ===")
        # 1. 加载工作流信息
        workflow_info = self._load_workflow_info(workflow_dir)
        
        result = RegenerationResult(
            workflow_id=workflow_info['id'],
            workflow_name=workflow_info['name'],
        )
        print(f"Workflow: {workflow_info['name']} (ID: {workflow_info['id']})")
        # 2. 发现工具
        tools_to_regenerate = self._discover_tools(workflow_dir)
        result.total_tools = len(tools_to_regenerate)
        
        if result.total_tools == 0:
            logger.warning(f"No tools found in workflow: {workflow_info['name']}")
            result.error_message = "No tools found"
            result.total_time = time.time() - start_time
            return result
        
        logger.info(f"Found {result.total_tools} tools to regenerate")
        
        # ============================================================
        # Phase 1: 生成所有工具 + 静态分析
        # ============================================================
        logger.info("=" * 60)
        logger.info("Phase 1: Tool Generation & Static Analysis")
        logger.info("=" * 60)
        
        for tool_name in tools_to_regenerate:
            original_path = self._find_original_tool(workflow_dir, tool_name)
            
            # 生成
            tool = await self.regenerate_tool(
                workflow_dir=workflow_dir,
                tool_name=tool_name,
                original_tool_path=original_path
            )
            
            # 保存代码
            tool.generated_path = self._save_generated_code(
                output_dir, tool_name, tool.code
            )
            
            # 静态分析
            tool.static_metrics = self._perform_static_analysis(tool)
            
            if tool.static_analysis_passed:
                result.static_analysis_passed += 1
                logger.info(f"  {tool_name}: ✅ Static analysis passed")
            else:
                logger.info(f"  {tool_name}: ❌ Static analysis failed")
            
            result.tools.append(tool)

        # ============================================================
        # Phase 2 & 3: 运行测试（使用 TestRunner 确保环境隔离）
        # ============================================================
        
        # 准备工具代码字典
        tools_code = {}
        for tool in result.tools:
            if tool.static_analysis_passed and tool.code:
                tools_code[tool.name] = tool.code
        
        # 创建测试运行器（传入 output_dir 用于保存调试日志）
        test_runner = TestRunner(
            workflow_dir=workflow_dir,
            timeout=self.config.e2e_config.execution_timeout,
            output_dir=output_dir  # 用于保存调试日志
        )
        
        # 运行所有测试（单工具 + 整体）
        print("\n=== Running Tests ===",tools_code)
        test_results = test_runner.run_all_tests(tools_code)
        
        print("\n=== Test Results Summary ===")

        print(f"Individual Test Results:", test_results)
        
        # 更新单工具测试结果
        logger.info("=" * 60)
        logger.info("Phase 2: Individual Tool Tests (Results)")
        logger.info("=" * 60)
        
        for tool in result.tools:
            if tool.name in test_results['individual_results']:
                indiv_result = test_results['individual_results'][tool.name]
                print(f"Tool: {tool.name}, Result: {indiv_result}")

                tool.individual_test = IndividualTestResult(
                    tool_name=tool.name,
                    success=indiv_result.get('success', False),
                    execution_time=indiv_result.get('execution_time', 0),
                    error_message=indiv_result.get('error'),
                    stdout=indiv_result.get('stdout'),
                    stderr=indiv_result.get('stderr'),
                )
                if tool.individual_test.success:
                    result.individual_test_passed += 1
            else:
                tool.individual_test = IndividualTestResult(
                    tool_name=tool.name,
                    success=False,
                    error_message="Skipped: static analysis failed"
                )
        
        # 更新整体测试结果
        logger.info("=" * 60)
        logger.info("Phase 3: Collective Workflow Test (Results)")
        logger.info("=" * 60)
        
        coll_result = test_results['collective_result']
        result.collective_test = CollectiveTestResult(
            success=coll_result.get('success', False),
            execution_time=coll_result.get('execution_time', 0),
            error_message=coll_result.get('error'),
            stdout=coll_result.get('stdout'),
            stderr=coll_result.get('stderr'),
            tools_tested=coll_result.get('tools_installed', []),
        )
        
        # ============================================================
        # Phase 4: 计算综合得分
        # ============================================================
        for tool in result.tools:
            score = self._calculate_overall_score(tool, result.collective_test)
            if score > 0.8:
                result.high_quality_tools += 1
        
        # ============================================================
        # Phase 5: 保存结果
        # ============================================================
        for tool in result.tools:
            self._save_tool_evaluation(output_dir, tool)
        
        result.total_time = time.time() - start_time
        if result.total_tools > 0:
            result.average_generation_time = sum(
                t.generation_time for t in result.tools
            ) / result.total_tools
        
        self._save_workflow_result(output_dir, result)
        
        # 打印总结
        self._print_result_summary(result)
        
        return result
    
    def _calculate_overall_score(
        self,
        tool: RegeneratedTool,
        collective_test: Optional[CollectiveTestResult]
    ) -> float:
        """
        计算工具的综合得分
        
        权重：
        - 静态分析: 30%
        - 单工具测试: 35%
        - 整体测试: 35%
        """
        score = 0.0
        
        # 静态分析 (30%)
        if tool.static_metrics:
            static_score = tool.static_metrics.get_overall_score()
            score += 0.30 * static_score
        
        # 单工具测试 (35%)
        if tool.individual_test and tool.individual_test.success:
            score += 0.35
        
        # 整体测试 (35%)
        if collective_test and collective_test.success:
            if tool.name in collective_test.tools_tested:
                score += 0.35
        
        return score
    
    def _print_result_summary(self, result: RegenerationResult):
        """打印结果总结"""
        print("\n" + "=" * 70)
        print(f"EVALUATION RESULT: {result.workflow_name}")
        print("=" * 70)
        
        # 工具统计
        print(f"\n📊 Tool Statistics:")
        print(f"   Total tools: {result.total_tools}")
        print(f"   Static analysis passed: {result.static_analysis_passed}/{result.total_tools} "
              f"({result.static_analysis_passed/result.total_tools*100:.1f}%)")
        
        # 单工具测试结果
        print(f"\n🔧 Individual Test Results (每个工具单独测试):")
        print(f"   Passed: {result.individual_test_passed}/{result.total_tools} "
              f"({result.individual_test_passed/result.total_tools*100:.1f}%)")
        for tool in result.tools:
            status = "✅" if tool.individual_test_success else "❌"
            print(f"   {status} {tool.name}")
        
        # 整体测试结果
        print(f"\n🔗 Collective Test Result (所有工具一起测试):")
        if result.collective_test:
            status = "✅ PASS" if result.collective_test.success else "❌ FAIL"
            print(f"   Status: {status}")
            print(f"   Tools tested: {', '.join(result.collective_test.tools_tested)}")
            if result.collective_test.error_message:
                print(f"   Error: {result.collective_test.error_message[:100]}")
        
        # 高质量工具
        print(f"\n⭐ High Quality Tools (score > 0.8):")
        print(f"   Count: {result.high_quality_tools}/{result.total_tools} "
              f"({result.high_quality_tools/result.total_tools*100:.1f}%)")
        
        # 时间
        print(f"\n⏱️  Time: {result.total_time:.2f}s total")
        
        print("=" * 70)
    
    # ========== 辅助方法 ==========
    
    def _load_workflow_info(self, workflow_dir: Path) -> Dict[str, str]:
        workflow_json = workflow_dir / "workflow.json"
        if workflow_json.exists():
            try:
                with open(workflow_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                workflow_info = data.get('workflow', {})
                return {
                    'id': workflow_info.get('name', workflow_dir.name).lower().replace(' ', '_'),
                    'name': workflow_info.get('name', workflow_dir.name),
                }
            except Exception as e:
                logger.warning(f"Failed to load workflow info: {e}")
        return {'id': workflow_dir.name, 'name': workflow_dir.name}
    
    def _discover_tools(self, workflow_dir: Path) -> List[str]:
        tools = []
        
        workflow_json = workflow_dir / "workflow.json"
        if workflow_json.exists():
            try:
                with open(workflow_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for step in data.get('workflow_steps', []):
                    tool_name = step.get('mcp_tool')
                    if tool_name and tool_name not in tools:
                        tools.append(tool_name)
            except Exception as e:
                logger.warning(f"Failed to discover tools: {e}")
        
        run_workflow = workflow_dir / "run_workflow.py"
        if run_workflow.exists():
            try:
                with open(run_workflow, 'r', encoding='utf-8') as f:
                    content = f.read()
                for match in re.finditer(r'call_tool\(\s*["\']([^"\']+)["\']', content):
                    tool_name = match.group(1)
                    if tool_name not in tools:
                        tools.append(tool_name)
            except Exception as e:
                logger.warning(f"Failed to discover tools: {e}")
        
        return tools
    
    def _find_original_tool(self, workflow_dir: Path, tool_name: str) -> Optional[Path]:
        """
        查找包含指定工具函数的原始文件
        
        支持：
        1. 单独文件：tool_name.py
        2. 合并文件：多个工具在同一个文件中
        """
        tools_dir = workflow_dir / "mcp_server" / "tools"
        if not tools_dir.exists():
            logger.warning(f"Tools directory not found: {tools_dir}")
            return None
        
        import ast
        
        # 1. 尝试精确文件名匹配
        exact_match = tools_dir / f"{tool_name}.py"
        if exact_match.exists():
            logger.info(f"Found tool in dedicated file: {exact_match}")
            return exact_match
        
        # 2. 扫描所有 Python 文件内容
        logger.debug(f"Scanning for '{tool_name}' in all .py files...")
        
        for tool_file in tools_dir.glob("*.py"):
            if tool_file.name == "__init__.py":
                continue
            
            try:
                with open(tool_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 使用 AST 查找函数定义
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name == tool_name:
                            logger.info(f"✓ Found '{tool_name}' in {tool_file.name}")
                            return tool_file
                        
                        # 模糊匹配（去掉 _tool 后缀）
                        normalized_func = node.name.replace('_tool', '').replace('_', '')
                        normalized_search = tool_name.replace('_tool', '').replace('_', '')
                        if normalized_func == normalized_search:
                            logger.info(f"✓ Found '{tool_name}' as '{node.name}' in {tool_file.name}")
                            return tool_file
            
            except Exception as e:
                logger.warning(f"Failed to scan {tool_file}: {e}")
        
        logger.warning(f"✗ Original tool '{tool_name}' not found in any file")
        return None

    def _save_generated_code(self, output_dir: Path, tool_name: str, code: str) -> Path:
        tools_dir = output_dir / "generated_tools"
        tools_dir.mkdir(exist_ok=True)
        tool_path = tools_dir / f"{tool_name}.py"
        with open(tool_path, 'w', encoding='utf-8') as f:
            f.write(code)
        return tool_path
    
    def _save_tool_evaluation(self, output_dir: Path, tool: RegeneratedTool):
        eval_dir = output_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True)

        metrics_file = eval_dir / f"{tool.name}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(tool.to_dict(), f, indent=2, ensure_ascii=False)

        if tool.individual_test:
            if tool.individual_test.stdout:
                with open(eval_dir / f"{tool.name}_individual_stdout.txt", 'w') as f:
                    f.write(tool.individual_test.stdout)
            if tool.individual_test.stderr:
                with open(eval_dir / f"{tool.name}_individual_stderr.txt", 'w') as f:
                    f.write(tool.individual_test.stderr)
    
    def _save_workflow_result(self, output_dir: Path, result: RegenerationResult):
        result_file = output_dir / "workflow_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        if result.collective_test:
            if result.collective_test.stdout:
                with open(output_dir / "collective_test_stdout.txt", 'w') as f:
                    f.write(result.collective_test.stdout)
            if result.collective_test.stderr:
                with open(output_dir / "collective_test_stderr.txt", 'w') as f:
                    f.write(result.collective_test.stderr)
        
        logger.info(f"Saved workflow result: {result_file}")