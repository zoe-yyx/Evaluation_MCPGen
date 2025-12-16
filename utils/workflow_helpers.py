"""
工作流评测辅助函数模块
包含评测中使用的各种计算和分析函数

迁移自: tasks/workflow_shuffling_mcp.py
"""

import re
from typing import Dict, List, Any


def calculate_lcs_similarity(seq1: List[str], seq2: List[str]) -> float:
    """
    计算两个序列的最长公共子序列(LCS)相似度
    
    Args:
        seq1: 第一个序列
        seq2: 第二个序列
    
    Returns:
        float: LCS相似度 (0.0 - 1.0)
    """
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


def calculate_position_errors(gen_order: List[str], ref_order: List[str]) -> Dict:
    """
    计算位置偏差
    
    Args:
        gen_order: 生成的顺序
        ref_order: 参考顺序
    
    Returns:
        Dict: 包含 mean, max, total 偏差信息
    """
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


def find_step_references(obj: Any) -> List[str]:
    """
    递归查找所有{{step_<id>.*}}引用
    
    Args:
        obj: 要搜索的对象 (dict, list, str)
    
    Returns:
        List[str]: 找到的步骤ID列表
    """
    refs = []
    if isinstance(obj, dict):
        for v in obj.values():
            refs.extend(find_step_references(v))
    elif isinstance(obj, list):
        for x in obj:
            refs.extend(find_step_references(x))
    elif isinstance(obj, str):
        refs.extend(re.findall(r'\{\{\s*step_(\w+)[^}]*\}\}', obj))
    return refs


def extract_linear_path(start_step: str, dependencies: Dict[str, List[str]]) -> List[str]:
    """
    从某个步骤开始，提取到终点的完整线性路径
    
    Args:
        start_step: 起始步骤ID
        dependencies: 依赖关系字典 {step_id: [next_step_ids]}
    
    Returns:
        List[str]: 线性路径
    """
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


def evaluate_dependency_satisfaction(ref_deps: Dict[str, List[str]], gen_order: List[str]) -> Dict:
    """
    评测依赖关系的满足度
    
    Args:
        ref_deps: 参考依赖关系
        gen_order: 生成的顺序
    
    Returns:
        Dict: 满足度评测结果
    """
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


def generate_summary_report(summary_results: Dict) -> str:
    """
    生成可读的汇总报告
    
    Args:
        summary_results: 汇总结果字典
    
    Returns:
        str: 格式化的报告文本
    """
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
            if isinstance(score, (int, float)):
                lines.append(f"Project {project_num}: ✅ SUCCESS (Score: {score:.2%}, Tools: {tools_count})")
            else:
                lines.append(f"Project {project_num}: ✅ SUCCESS (Score: {score}, Tools: {tools_count})")
        elif status == 'skipped':
            reason = detail.get('reason', 'Unknown')
            lines.append(f"Project {project_num}: ⚠️ SKIPPED ({reason})")
        else:
            error = detail.get('error', 'Unknown error')
            lines.append(f"Project {project_num}: ❌ FAILED ({error[:50]}...)")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def check_has_branches(steps: List[Dict]) -> bool:
    """
    检查工作流是否有分支结构
    
    Args:
        steps: 工作流步骤列表
    
    Returns:
        bool: 是否有分支
    """
    for step in steps:
        next_steps = step.get('next_steps', [])
        if len(next_steps) > 1:
            return True
    return False


def extract_dependencies(steps: List[Dict]) -> Dict[str, List[str]]:
    """
    提取步骤间的依赖关系（next_steps）
    
    Args:
        steps: 工作流步骤列表
    
    Returns:
        Dict[str, List[str]]: 依赖关系字典
    """
    deps = {}
    for step in steps:
        if step.get('next_steps'):
            deps[step['step_id']] = step['next_steps']
    return deps


def extract_dataflow(steps: List[Dict]) -> Dict[str, List[str]]:
    """
    提取数据流依赖（{{step_X.output}}引用）
    
    Args:
        steps: 工作流步骤列表
    
    Returns:
        Dict[str, List[str]]: 数据流依赖字典
    """
    dataflow = {}
    for step in steps:
        refs = find_step_references(step.get('parameters', {}))
        refs.extend(find_step_references(step.get('condition', '')))
        if refs:
            dataflow[step['step_id']] = list(set(refs))
    return dataflow


def extract_control_flow(steps: List[Dict]) -> Dict[str, Dict]:
    """
    提取控制流信息（else_steps, error_handler）
    
    Args:
        steps: 工作流步骤列表
    
    Returns:
        Dict[str, Dict]: 控制流信息字典
    """
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
