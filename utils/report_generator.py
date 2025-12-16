"""
评测报告生成模块
包含评测报告的生成和格式化函数

迁移自: tasks/workflow_shuffling_mcp.py
"""

from typing import Dict, List, Any
from pathlib import Path

from .workflow_helpers import extract_linear_path


def generate_evaluation_report(
    results: Dict,
    mcp_tools: List[Any],
    reference_answer: Dict,
    generated_workflow_steps: List[Dict],
    output_path: Path = None
) -> str:
    """
    生成详细评测报告
    
    Args:
        results: 评测结果字典
        mcp_tools: MCP工具列表
        reference_answer: 参考答案
        generated_workflow_steps: 生成的工作流步骤
        output_path: 输出文件路径（可选）
    
    Returns:
        str: 报告文本
    """
    lines = [
        "=" * 80,
        "Workflow Orchestration Evaluation Report (MCP Version)",
        "=" * 80,
        f"Timestamp: {results['timestamp']}",
        f"MCP Tools Used: {len(mcp_tools)} tools",
        "",
    ]
    
    # 显示使用的MCP工具
    if mcp_tools:
        lines.append("📦 MCP Tools:")
        for tool in mcp_tools:
            desc = tool.description if hasattr(tool, 'description') else str(tool.get('description', ''))
            name = tool.name if hasattr(tool, 'name') else str(tool.get('name', ''))
            lines.append(f"   - {name}: {desc[:60]}...")
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

    err_metrics = results['metrics']['error_propagation']
    lines.extend([
        "Error Propagation Rate",
        f"- Propagation Rate: {err_metrics['error_propagation_rate']:.2%}",
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
    if generated_workflow_steps:
        for step in generated_workflow_steps:
            if step.get('next_steps'):
                gen_deps[step['step_id']] = step['next_steps']
    
    if reference_answer.get('has_branches'):
        lines.append("📋 Reference Structure:")
        ref_deps = reference_answer['dependencies']
        
        branch_points = [(k, v) for k, v in ref_deps.items() if len(v) > 1]
        if branch_points:
            branch_point_id, branch_targets = branch_points[0]
            branch_index = reference_answer['execution_order'].index(branch_point_id)
            main_path = reference_answer['execution_order'][:branch_index + 1]
            
            lines.append(f"  Main Path: {' -> '.join(main_path)}")
            lines.append(f"  Branches from {branch_point_id}:")
            
            for i, branch_id in enumerate(branch_targets, 1):
                branch_path = extract_linear_path(branch_id, ref_deps)
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
                branch_path = extract_linear_path(branch_id, gen_deps)
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
        lines.append("✅ EXCELLENT")
    elif results['overall_score'] >= 0.80:
        lines.append("✅ PASSED")
    elif results['overall_score'] >= 0.60:
        lines.append("⚠️  NEEDS IMPROVEMENT")
    else:
        lines.append("❌ FAILED")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report
