"""
tasks/comprehensive_metrics.py

"""
import ast
import importlib.util
import sys
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveToolMetrics:
    """综合工具评测指标"""
    
    # === 1. 代码质量指标 (Code Quality) ===
    syntax_valid: bool = False
    syntax_errors: List[str] = field(default_factory=list)
    
    imports_valid: bool = False
    import_errors: List[str] = field(default_factory=list)
    missing_modules: List[str] = field(default_factory=list)
    
    has_target_function: bool = False
    target_function_name: Optional[str] = None
    
    has_docstring: bool = False
    docstring_quality_score: float = 0.0  # 0-1
    
    follows_naming_conventions: bool = False
    naming_violations: List[str] = field(default_factory=list)
    
    has_type_hints: bool = False
    type_coverage: float = 0.0  # 参数类型覆盖率
    
    no_dangerous_operations: bool = True
    dangerous_operations: List[str] = field(default_factory=list)
    
    # === 2. 签名匹配指标 (Signature Matching) ===
    signature_matches_original: bool = False
    param_count_correct: bool = False
    param_names_match: bool = False
    param_types_match: bool = False
    return_type_matches: bool = False
    
    original_signature: Optional[str] = None
    generated_signature: Optional[str] = None
    signature_diff: Optional[str] = None
    
    # === 3. 功能正确性指标 (Functional Correctness) ===
    workflow_execution_success: bool = False
    execution_error_message: Optional[str] = None
    execution_time: float = 0.0
    
    output_format_correct: bool = False
    output_validation_passed: bool = False
    
    test_stdout: Optional[str] = None
    test_stderr: Optional[str] = None
    
    # === 4. 代码相似度指标 (Code Similarity) ===
    code_similarity_score: float = 0.0  # 0-1, 与原代码的相似度
    structural_similarity: float = 0.0  # AST 结构相似度
    
    # === 5. 额外信息 ===
    warnings: List[str] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_overall_score(self) -> float:
        """
        计算综合得分 (0-1)
        
        权重分配：
        - 代码质量: 30%
        - 签名匹配: 25%
        - 功能正确性: 35%
        - 代码规范: 10%
        """
        weights = {
            'syntax_valid': 0.10,
            'imports_valid': 0.08,
            'has_target_function': 0.12,
            'signature_matches_original': 0.25,
            'workflow_execution_success': 0.30,
            'has_docstring': 0.03,
            'has_type_hints': 0.05,
            'follows_naming_conventions': 0.02,
            'no_dangerous_operations': 0.05,
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = getattr(self, metric, False)
            if isinstance(value, bool):
                score += weight if value else 0
            elif isinstance(value, (int, float)):
                score += weight * min(1.0, max(0.0, value))
        
        return score
    
    def get_category_scores(self) -> Dict[str, float]:
        """获取各类别得分"""
        return {
            'code_quality': self._calc_code_quality_score(),
            'signature_matching': self._calc_signature_score(),
            'functional_correctness': self._calc_functional_score(),
            'code_standards': self._calc_standards_score(),
        }
    
    def _calc_code_quality_score(self) -> float:
        score = 0.0
        if self.syntax_valid: score += 0.3
        if self.imports_valid: score += 0.25
        if self.has_target_function: score += 0.35
        if self.no_dangerous_operations: score += 0.1
        return score
    
    def _calc_signature_score(self) -> float:
        if not self.has_target_function:
            return 0.0
        score = 0.0
        if self.param_count_correct: score += 0.4
        if self.param_names_match: score += 0.3
        if self.param_types_match: score += 0.2
        if self.return_type_matches: score += 0.1
        return score
    
    def _calc_functional_score(self) -> float:
        score = 0.0
        if self.workflow_execution_success: score += 0.7
        if self.output_format_correct: score += 0.15
        if self.output_validation_passed: score += 0.15
        return score
    
    def _calc_standards_score(self) -> float:
        score = 0.0
        if self.has_docstring: score += 0.3
        if self.has_type_hints: score += 0.4
        if self.follows_naming_conventions: score += 0.3
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于序列化"""
        return {
            'overall_score': self.get_overall_score(),
            'category_scores': self.get_category_scores(),
            
            # 代码质量
            'code_quality': {
                'syntax_valid': self.syntax_valid,
                'syntax_errors': self.syntax_errors,
                'imports_valid': self.imports_valid,
                'import_errors': self.import_errors,
                'missing_modules': self.missing_modules,
                'has_target_function': self.has_target_function,
                'has_docstring': self.has_docstring,
                'docstring_quality_score': self.docstring_quality_score,
                'has_type_hints': self.has_type_hints,
                'type_coverage': self.type_coverage,
                'follows_naming_conventions': self.follows_naming_conventions,
                'naming_violations': self.naming_violations,
                'no_dangerous_operations': self.no_dangerous_operations,
                'dangerous_operations': self.dangerous_operations,
            },
            
            # 签名匹配
            'signature_matching': {
                'matches_original': self.signature_matches_original,
                'param_count_correct': self.param_count_correct,
                'param_names_match': self.param_names_match,
                'param_types_match': self.param_types_match,
                'return_type_matches': self.return_type_matches,
                'original_signature': self.original_signature,
                'generated_signature': self.generated_signature,
                'signature_diff': self.signature_diff,
            },
            
            # 功能正确性
            'functional_correctness': {
                'workflow_execution_success': self.workflow_execution_success,
                'execution_error_message': self.execution_error_message,
                'execution_time': self.execution_time,
                'output_format_correct': self.output_format_correct,
                'output_validation_passed': self.output_validation_passed,
            },
            
            # 代码相似度
            'similarity': {
                'code_similarity_score': self.code_similarity_score,
                'structural_similarity': self.structural_similarity,
            },
            
            # 其他
            'warnings': self.warnings,
            'metadata': self.analysis_metadata,
        }


class ComprehensiveMetricsAnalyzer:
    """综合指标分析器"""
    
    def __init__(self, original_tool_path: Optional[Path] = None):
        self.original_tool_path = original_tool_path
        self.original_tool_info = None
        
        if original_tool_path and original_tool_path.exists():
            self.original_tool_info = self._extract_original_info()
    
    def analyze(
        self, 
        generated_code: str, 
        tool_name: str,
        execution_result: Optional[Dict[str, Any]] = None
    ) -> ComprehensiveToolMetrics:
        """
        执行综合分析
        
        Args:
            generated_code: 生成的代码
            tool_name: 工具名称
            execution_result: 可选的执行结果 (包含 success, stdout, stderr 等)
        
        Returns:
            ComprehensiveToolMetrics
        """
        metrics = ComprehensiveToolMetrics()
        metrics.target_function_name = tool_name
        
        # 1. 语法检查
        self._check_syntax(generated_code, metrics)
        if not metrics.syntax_valid:
            return metrics  # 语法错误就不继续了
        
        # 2. AST 分析
        try:
            tree = ast.parse(generated_code)
            
            # 2.1 导入检查
            self._check_imports(tree, metrics)
            
            # 2.2 查找目标函数
            func_node = self._find_function(tree, tool_name)

            if func_node:
                metrics.has_target_function = True
                
                # 2.3 文档字符串
                self._check_docstring(func_node, metrics)
                
                # 2.4 类型提示
                self._check_type_hints(func_node, metrics)
                
                # 2.5 命名规范
                self._check_naming_conventions(func_node, metrics)
                
                # 2.6 签名对比（如果有原工具）
                if self.original_tool_info:
                    self._compare_signatures(func_node, metrics)
            else:
                metrics.syntax_errors.append(f"Function '{tool_name}' not found")
            
            # 2.7 危险操作检查
            self._check_dangerous_operations(tree, metrics)
            
        except Exception as e:
            metrics.syntax_errors.append(f"AST analysis failed: {str(e)}")
            return metrics
        
        # 3. 代码相似度（如果有原工具）
        if self.original_tool_path and self.original_tool_path.exists():
            self._calculate_similarity(generated_code, metrics)
        
        # 4. 功能正确性（如果提供了执行结果）
        if execution_result:
            self._analyze_execution(execution_result, metrics)
        
        return metrics
    
    def _check_syntax(self, code: str, metrics: ComprehensiveToolMetrics):
        """检查语法"""
        try:
            compile(code, "<string>", "exec")
            metrics.syntax_valid = True
        except SyntaxError as e:
            metrics.syntax_valid = False
            metrics.syntax_errors.append(
                f"Line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            metrics.syntax_valid = False
            metrics.syntax_errors.append(f"Compilation error: {str(e)}")
    
    def _check_imports(self, tree: ast.AST, metrics: ComprehensiveToolMetrics):
        """检查导入"""
        all_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    all_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    all_imports.append(node.module)
        
        # 检查每个导入是否可用
        missing = []
        for module_name in all_imports:
            if not self._is_module_available(module_name):
                missing.append(module_name)
                metrics.import_errors.append(f"Module '{module_name}' not available")
        
        metrics.missing_modules = missing
        metrics.imports_valid = len(missing) == 0
    
    def _is_module_available(self, module_name: str) -> bool:
        """检查模块是否可用"""
        # 分割子模块（如 os.path）
        base_module = module_name.split('.')[0]
        
        try:
            # 标准库检查
            if hasattr(sys, 'stdlib_module_names') and base_module in sys.stdlib_module_names:
                return True
            
            # 尝试查找
            spec = importlib.util.find_spec(base_module)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
            return False
    
    def _find_function(self, tree: ast.AST, function_name: str) -> Optional[ast.FunctionDef]:
        """查找函数定义（支持同步、异步和模糊匹配）"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    return node
        
        normalized_search = function_name.replace('_tool', '').replace('_', '').lower()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                normalized_func = node.name.replace('_tool', '').replace('_', '').lower()
                if normalized_func == normalized_search:
                    logger.warning(
                        f"Found function with different name: "
                        f"expected '{function_name}', got '{node.name}'"
                    )
                    return node
        return None
    
    def _check_docstring(self, func: ast.FunctionDef, metrics: ComprehensiveToolMetrics):
        """检查文档字符串"""
        docstring = ast.get_docstring(func)
        metrics.has_docstring = docstring is not None
        
        if docstring:
            # 简单的质量评分
            score = 0.0
            if len(docstring) > 20: score += 0.3  # 足够长
            if 'Args:' in docstring or 'Parameters:' in docstring: score += 0.3  # 有参数说明
            if 'Returns:' in docstring or 'Return:' in docstring: score += 0.2  # 有返回值说明
            if len(docstring.split('\n')) > 2: score += 0.2  # 多行
            metrics.docstring_quality_score = min(1.0, score)
        else:
            metrics.warnings.append("Missing docstring")
    
    def _check_type_hints(self, func: ast.FunctionDef, metrics: ComprehensiveToolMetrics):
        """检查类型提示"""
        total_params = len(func.args.args)
        typed_params = sum(1 for arg in func.args.args if arg.annotation is not None)
        
        metrics.has_type_hints = typed_params > 0
        metrics.type_coverage = typed_params / total_params if total_params > 0 else 0.0
        
        if typed_params < total_params:
            metrics.warnings.append(
                f"Only {typed_params}/{total_params} parameters have type hints"
            )
        
        if func.returns is None:
            metrics.warnings.append("Missing return type hint")
    
    def _check_naming_conventions(self, func: ast.FunctionDef, metrics: ComprehensiveToolMetrics):
        """检查命名规范 (PEP 8)"""
        violations = []
        
        # 函数名应该是 snake_case
        if not func.name.islower():
            violations.append(f"Function name '{func.name}' should be lowercase")
        
        if func.name.startswith('_') and not func.name.startswith('__'):
            # 私有函数，这里假设工具函数不应该是私有的
            violations.append(f"Tool function should not start with '_'")
        
        # 检查参数名
        for arg in func.args.args:
            if not arg.arg.islower() or arg.arg.startswith('_'):
                violations.append(f"Parameter '{arg.arg}' should be lowercase")
        
        metrics.naming_violations = violations
        metrics.follows_naming_conventions = len(violations) == 0
    
    def _check_dangerous_operations(self, tree: ast.AST, metrics: ComprehensiveToolMetrics):
        """检查危险操作"""
        dangerous_funcs = {'eval', 'exec', 'compile', '__import__', 'open'}
        dangerous_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_funcs:
                        dangerous_found.append(node.func.id)
        
        metrics.dangerous_operations = list(set(dangerous_found))
        metrics.no_dangerous_operations = len(dangerous_found) == 0
        
        if dangerous_found:
            metrics.warnings.append(
                f"Dangerous operations found: {', '.join(set(dangerous_found))}"
            )
    
    def _compare_signatures(self, func: ast.FunctionDef, metrics: ComprehensiveToolMetrics):
        """对比函数签名"""
        if not self.original_tool_info:
            return
        
        original = self.original_tool_info
        
        # 生成的参数
        gen_params = [arg.arg for arg in func.args.args]
        
        # 原始参数
        orig_params = original.get('parameters', [])
        
        # 对比
        metrics.param_count_correct = len(gen_params) == len(orig_params)
        metrics.param_names_match = gen_params == orig_params
        
        # TODO: 更复杂的类型对比
        metrics.param_types_match = True  # 简化
        metrics.return_type_matches = True  # 简化
        
        metrics.signature_matches_original = (
            metrics.param_count_correct and metrics.param_names_match
        )
        
        # 记录签名信息
        metrics.original_signature = f"{original.get('name')}({', '.join(orig_params)})"
        metrics.generated_signature = f"{func.name}({', '.join(gen_params)})"
        
        if not metrics.signature_matches_original:
            metrics.signature_diff = (
                f"Expected: {metrics.original_signature}\n"
                f"Got: {metrics.generated_signature}"
            )
    
    def _calculate_similarity(self, generated_code: str, metrics: ComprehensiveToolMetrics):
        """计算代码相似度"""
        try:
            with open(self.original_tool_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # 字符串相似度
            matcher = difflib.SequenceMatcher(None, original_code, generated_code)
            metrics.code_similarity_score = matcher.ratio()
            
            # TODO: AST 结构相似度
            metrics.structural_similarity = 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate similarity: {e}")
    
    def _analyze_execution(self, execution_result: Dict[str, Any], metrics: ComprehensiveToolMetrics):
        """分析执行结果"""
        metrics.workflow_execution_success = execution_result.get('success', False)
        metrics.execution_time = execution_result.get('execution_time', 0.0)
        metrics.execution_error_message = execution_result.get('error')
        metrics.test_stdout = execution_result.get('stdout')
        metrics.test_stderr = execution_result.get('stderr')
        
        # 输出格式检查（简单版本）
        if metrics.workflow_execution_success:
            metrics.output_format_correct = True
            metrics.output_validation_passed = True
        else:
            # 检查错误类型
            if metrics.test_stderr:
                stderr = metrics.test_stderr.lower()
                if 'importerror' in stderr or 'modulenotfounderror' in stderr:
                    metrics.warnings.append("Import error during execution")
                elif 'syntaxerror' in stderr:
                    metrics.warnings.append("Syntax error during execution")
                elif 'typeerror' in stderr:
                    metrics.warnings.append("Type error during execution")
    
    def _extract_original_info(self) -> Dict[str, Any]:
        """提取原工具信息"""
        try:
            with open(self.original_tool_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            tree = ast.parse(code)
            func_name = self.original_tool_path.stem
            func_node = self._find_function(tree, func_name)
            
            if not func_node:
                return {}
            
            return {
                'name': func_name,
                'parameters': [arg.arg for arg in func_node.args.args],
                'has_return_type': func_node.returns is not None,
                'docstring': ast.get_docstring(func_node),
            }
        except Exception as e:
            logger.warning(f"Failed to extract original tool info: {e}")
            return {}