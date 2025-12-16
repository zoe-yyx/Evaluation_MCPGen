"""
错误传播率评测模块 - 集成补丁
=====================================

将此模块添加到 utils/ 目录下，文件名: error_propagation.py

然后在 workflow_evaluation.py 中进行以下修改：

1. 在文件顶部添加导入：
   from utils.error_propagation import calculate_error_propagation

2. 在 evaluate() 方法中添加错误传播率评测（约第150行后）：
   results['metrics']['error_propagation'] = self._evaluate_error_propagation(gen_steps)

3. 添加 _evaluate_error_propagation 方法

4. 更新 _calculate_overall_score 方法的权重

详细修改见下方代码块标注。
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque


# ==================== 核心计算函数 ====================

def build_dependency_graph(dependencies: Dict[str, List[str]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    从依赖关系构建正向和反向图
    
    Args:
        dependencies: {step_id: [next_step_ids]} 格式的依赖关系
    
    Returns:
        forward_graph: {node: set(children)} - 正向图（父->子）
        backward_graph: {node: set(parents)} - 反向图（子->父）
    """
    forward_graph = defaultdict(set)
    backward_graph = defaultdict(set)
    all_nodes = set()
    
    for parent, children in dependencies.items():
        all_nodes.add(parent)
        for child in children:
            all_nodes.add(child)
            forward_graph[parent].add(child)
            backward_graph[child].add(parent)
    
    for node in all_nodes:
        if node not in forward_graph:
            forward_graph[node] = set()
        if node not in backward_graph:
            backward_graph[node] = set()
    
    return dict(forward_graph), dict(backward_graph)


def get_all_descendants(node: str, forward_graph: Dict[str, Set[str]]) -> Set[str]:
    """获取节点的所有下游节点（BFS遍历）"""
    descendants = set()
    queue = deque([node])
    visited = {node}
    
    while queue:
        current = queue.popleft()
        for child in forward_graph.get(current, []):
            if child not in visited:
                visited.add(child)
                descendants.add(child)
                queue.append(child)
    
    return descendants


def get_longest_path_from_node(node: str, forward_graph: Dict[str, Set[str]]) -> int:
    """计算从节点出发的最长路径长度（DFS + 记忆化）"""
    memo = {}
    
    def dfs(n: str) -> int:
        if n in memo:
            return memo[n]
        
        children = forward_graph.get(n, set())
        if not children:
            memo[n] = 0
            return 0
        
        max_child_path = max(dfs(child) for child in children)
        memo[n] = 1 + max_child_path
        return memo[n]
    
    return dfs(node)


def identify_error_nodes(
    ref_order: List[str],
    gen_order: List[str],
    ref_deps: Dict[str, List[str]],
    gen_deps: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """
    识别生成工作流中的错误节点
    
    错误类型：
    1. missing: 参考中有但生成中没有的节点
    2. extra: 生成中有但参考中没有的节点
    3. order_or_dependency: 位置或依赖关系错误
    """
    error_nodes = {}
    
    ref_set = set(ref_order)
    gen_set = set(gen_order)
    
    # 缺失节点
    for node in (ref_set - gen_set):
        error_nodes[node] = {
            'error_type': 'missing',
            'severity': 1.0,
            'details': 'Node missing in generated workflow'
        }
    
    # 多余节点
    for node in (gen_set - ref_set):
        error_nodes[node] = {
            'error_type': 'extra',
            'severity': 0.5,
            'details': 'Node not in reference workflow'
        }
    
    # 位置和依赖错误
    common_nodes = ref_set & gen_set
    ref_pos = {node: i for i, node in enumerate(ref_order)}
    gen_pos = {node: i for i, node in enumerate(gen_order)}
    
    for node in common_nodes:
        errors = []
        
        # 检查位置错误
        ref_p, gen_p = ref_pos.get(node, -1), gen_pos.get(node, -1)
        if ref_p != gen_p:
            position_shift = abs(ref_p - gen_p) / len(ref_order)
            errors.append({
                'type': 'position',
                'ref_position': ref_p,
                'gen_position': gen_p,
                'shift': position_shift
            })
        
        # 检查依赖错误
        ref_next = set(ref_deps.get(node, []))
        gen_next = set(gen_deps.get(node, []))
        
        if ref_next != gen_next:
            missing_deps = ref_next - gen_next
            extra_deps = gen_next - ref_next
            dep_error_rate = len(missing_deps | extra_deps) / max(len(ref_next), 1)
            errors.append({
                'type': 'dependency',
                'missing_deps': list(missing_deps),
                'extra_deps': list(extra_deps),
                'error_rate': dep_error_rate
            })
        
        if errors:
            severity = sum(
                e['shift'] * 0.3 if e['type'] == 'position' else e['error_rate'] * 0.7
                for e in errors
            )
            error_nodes[node] = {
                'error_type': 'order_or_dependency',
                'severity': min(severity, 1.0),
                'details': errors
            }
    
    return error_nodes


def calculate_error_propagation(
    ref_order: List[str],
    gen_order: List[str],
    ref_deps: Dict[str, List[str]],
    gen_steps: List[Dict]
) -> Dict:
    """
    计算错误传播率
    
    核心指标：
    - score: 综合得分 (0-1, 1=无错误传播)
    - error_propagation_rate: 错误传播率 (0-1, 0=无传播)
    - propagation_depth: 最大传播深度
    - propagation_breadth: 受影响节点比例
    - weighted_impact: 加权影响（考虑错误位置）
    """
    # 构建生成工作流的依赖关系
    gen_deps = {
        step['step_id']: step['next_steps']
        for step in gen_steps if step.get('next_steps')
    }
    
    # 构建参考工作流的依赖图
    forward_graph, _ = build_dependency_graph(ref_deps)
    
    # 识别错误节点
    error_nodes = identify_error_nodes(ref_order, gen_order, ref_deps, gen_deps)
    
    if not error_nodes:
        return {
            'score': 1.0,
            'error_propagation_rate': 0.0,
            'propagation_depth': 0,
            'propagation_breadth': 0.0,
            'weighted_impact': 0.0,
            'error_nodes_count': 0,
            'affected_nodes_count': 0,
            'total_nodes': len(ref_order),
            'details': {'error_nodes': {}, 'propagation_chains': []}
        }
    
    total_nodes = len(ref_order)
    ref_pos = {node: i for i, node in enumerate(ref_order)}
    
    propagation_chains = []
    total_weighted_impact = 0.0
    max_propagation_depth = 0
    all_affected_nodes = set()
    
    for error_node, error_info in error_nodes.items():
        if error_node not in forward_graph:
            continue
        
        descendants = get_all_descendants(error_node, forward_graph)
        all_affected_nodes.update(descendants)
        
        propagation_depth = get_longest_path_from_node(error_node, forward_graph)
        max_propagation_depth = max(max_propagation_depth, propagation_depth)
        
        node_position = ref_pos.get(error_node, total_nodes - 1)
        position_weight = 1 - (node_position / total_nodes)
        
        breadth_impact = len(descendants) / total_nodes if total_nodes > 0 else 0
        depth_impact = propagation_depth / total_nodes if total_nodes > 0 else 0
        
        weighted_impact = error_info['severity'] * position_weight * (0.6 * breadth_impact + 0.4 * depth_impact)
        total_weighted_impact += weighted_impact
        
        propagation_chains.append({
            'error_node': error_node,
            'error_type': error_info['error_type'],
            'severity': error_info['severity'],
            'position': node_position,
            'position_weight': position_weight,
            'descendants': list(descendants),
            'descendant_count': len(descendants),
            'propagation_depth': propagation_depth,
            'breadth_impact': breadth_impact,
            'depth_impact': depth_impact,
            'weighted_impact': weighted_impact
        })
    
    propagation_breadth = len(all_affected_nodes) / total_nodes if total_nodes > 0 else 0
    normalized_impact = total_weighted_impact / len(error_nodes) if error_nodes else 0
    error_propagation_rate = min(1.0, propagation_breadth * 0.5 + normalized_impact * 0.5)
    score = 1.0 - error_propagation_rate
    
    return {
        'score': score,
        'error_propagation_rate': error_propagation_rate,
        'propagation_depth': max_propagation_depth,
        'propagation_breadth': propagation_breadth,
        'weighted_impact': normalized_impact,
        'error_nodes_count': len(error_nodes),
        'affected_nodes_count': len(all_affected_nodes),
        'total_nodes': total_nodes,
        'details': {
            'error_nodes': error_nodes,
            'propagation_chains': propagation_chains
        }
    }


# ==================== 集成说明 ====================
"""
将以下代码添加到 WorkflowEvaluationSystem 类中：

# ---------- 在 evaluate() 方法中添加（约第155行）----------

    def evaluate(self, generated_workflow: Dict) -> Dict:
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
        
        # 4. 错误传播率 [NEW]
        results['metrics']['error_propagation'] = self._evaluate_error_propagation(gen_steps)
        
        # 计算总分
        results['overall_score'] = self._calculate_overall_score(results['metrics'])
        
        return results


# ---------- 添加新方法 ----------

    def _evaluate_error_propagation(self, gen_steps: List[Dict]) -> Dict:
        '''评测错误传播率'''
        from utils.error_propagation import calculate_error_propagation
        
        return calculate_error_propagation(
            ref_order=self.reference_answer['execution_order'],
            gen_order=[s['step_id'] for s in gen_steps],
            ref_deps=self.reference_answer['dependencies'],
            gen_steps=gen_steps
        )


# ---------- 更新 _calculate_overall_score 方法 ----------

    def _calculate_overall_score(self, metrics: Dict) -> float:
        '''计算总分（更新权重）'''
        order_metrics = metrics.get('order_accuracy', {})
        order_score = order_metrics.get('overall_path_correctness', 0.0)
        
        # 更新后的权重分配
        weights = {
            'order': 0.40,                    # 从0.50降至0.40
            'dependency_accuracy': 0.20,      # 从0.25降至0.20
            'control_flow_accuracy': 0.20,    # 从0.25降至0.20
            'error_propagation': 0.20,        # 新增
        }
        
        total_score = (
            weights['order'] * order_score +
            weights['dependency_accuracy'] * metrics.get('dependency_accuracy', {}).get('score', 0.0) +
            weights['control_flow_accuracy'] * metrics.get('control_flow_accuracy', {}).get('score', 0.0) +
            weights['error_propagation'] * metrics.get('error_propagation', {}).get('score', 0.0)
        )
        
        return total_score
"""