"""
Unified Workflow Orchestration Evaluation System
统一的工作流编排评测系统 - 支持分支结构的多维度评测

核心功能：
1. 自动检测工作流类型（线性/分支）
2. ID混淆和文本清理
3. 多维度评测（顺序、依赖、数据流、控制流、拓扑）
4. 详细的评测报告
"""

import json
import random
import copy
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import sys
import re
import bisect
import asyncio
from tqdm import tqdm

# 假设这些模块存在
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from utils.llm_interface import LLMInterface
from core.config import LLMConfig
from utils.evaluate_tools import obfuscate_steps, parse_llm_output

# 混淆工具已移至 `utils.evaluate_tools.obfuscate_steps`

# ==================== 主评测系统 ====================

class WorkflowEvaluationSystem:
    """统一的工作流评测系统"""
    
    def __init__(self, reference_workflow_path: str, seed: int = 42, obfuscate: bool = True):
        self.reference_workflow = self._load_workflow(reference_workflow_path)
        self.seed = seed
        self.obfuscate = obfuscate
        random.seed(seed)
        
        # 提取参考答案
        self.reference_answer = self._extract_reference_answer()
        self.id_mapping = None  # 存储混淆后的ID映射
        self.generated_workflow_steps = None  # 存储生成的工作流步骤
    
    def _load_workflow(self, path: str) -> Dict:
        """加载参考工作流"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_reference_answer(self) -> Dict:
        """
        从参考工作流中提取评测所需的所有信息
        """
        steps = self.reference_workflow['workflow_steps']
        
        return {
            'execution_order': [s['step_id'] for s in steps],  # 保留原始顺序用于显示
            'steps_detail': {s['step_id']: s for s in steps},
            'dependencies': self._extract_dependencies(steps),
            'dataflow': self._extract_dataflow(steps),
            'control_flow': self._extract_control_flow(steps),
            'step_count': len(steps),
            'has_branches': self._check_has_branches(steps)  # 新增：检查是否有分支
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
            # 匹配 {{step_<id>...}} 或 {{ step_<id>... }}
            refs.extend(re.findall(r'\{\{\s*step_(\w+)[^}]*\}\}', obj))
        return refs
    
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
        
    def evaluate(self, generated_workflow: Dict) -> Dict:
        """
        多维度评测生成的工作流
        
        评测维度：
        1. 执行顺序准确性 (Execution Order Accuracy)
        2. 依赖关系准确性 (Dependency Accuracy) 
        3. 数据流有效性 (Dataflow Validity)
        4. 控制流准确性 (Control Flow Accuracy)
        5. 拓扑有效性 (Topology Validity)
        """
        gen_steps = generated_workflow['workflow_steps']
        gen_order = [s['step_id'] for s in gen_steps]
        
        # 保存生成的步骤信息，供后续使用
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
        
        # 3. 数据流有效性
        # results['metrics']['dataflow_validity'] = self._evaluate_dataflow(gen_steps)
        
        # 4. 控制流准确性
        results['metrics']['control_flow_accuracy'] = self._evaluate_control_flow(gen_steps)
        
        # 5. 拓扑有效性
        # results['metrics']['topology_validity'] = self._evaluate_topology(gen_steps)
        
        # 计算总分
        results['overall_score'] = self._calculate_overall_score(results['metrics'])
        
        return results
    
    def _evaluate_order(self, generated_order: List[str]) -> Dict:
        """
        评测执行顺序准确性
        - 线性工作流：严格序列匹配
        - 分支工作流：分支结构匹配 + 线性序列验证
        """
        ref_order = self.reference_answer['execution_order']
        ref_deps = self.reference_answer['dependencies']
        
        # 检查是否有分支结构
        has_branches = any(len(nexts) > 1 for nexts in ref_deps.values())
        
        if has_branches:
            # 对于分支结构，使用分支工作流评测
            return self._evaluate_branching_workflow(self.generated_workflow_steps)
        else:
            # 对于线性结构，使用严格的序列匹配
            return self._evaluate_linear_order(generated_order, ref_order)
    
    def _evaluate_branching_workflow(self, gen_steps: List[Dict]) -> Dict:
        """
        针对分支工作流的评测
        返回4个独立指标：
        1. 整体路径正确性 (overall_path_correctness)
        2. 主路径准确度 (main_path_accuracy)
        3. 分支结构正确性 (branch_structure_correctness)
        4. 分支路径准确度 (branch_path_accuracy)
        """
        ref_deps = self.reference_answer['dependencies']
        ref_order = self.reference_answer['execution_order']
        
        # 构建生成工作流的依赖图
        gen_deps = {}
        for step in gen_steps:
            if step.get('next_steps'):
                gen_deps[step['step_id']] = step['next_steps']
        
        gen_order = [s['step_id'] for s in gen_steps]
        
        results = {
            'evaluation_type': 'branching_workflow'
        }
        
        # 1. 识别参考答案的所有分支点
        ref_branch_points = [(k, v) for k, v in ref_deps.items() if len(v) > 1]
        
        if not ref_branch_points:
            return self._evaluate_linear_order(gen_order, ref_order)
        
        # 假设只有一个主分支点
        first_branch_point_id, ref_branch_targets = ref_branch_points[0]
        
        # ========== 指标1: 主路径准确度 ==========
        main_path_result = self._evaluate_main_path(
            first_branch_point_id, 
            ref_order, 
            gen_order
        )
        results['main_path_accuracy'] = main_path_result['accuracy']
        results['main_path_details'] = main_path_result
        
        # ========== 指标2: 分支结构正确性 ==========
        branch_structure_result = self._evaluate_branch_structure(
            first_branch_point_id,
            ref_branch_targets,
            gen_deps
        )
        results['branch_structure_correctness'] = 1.0 if branch_structure_result['correct'] else 0.0
        results['branch_structure_details'] = branch_structure_result
        
        # ========== 指标3: 分支路径准确度 ==========
        branch_paths_result = self._evaluate_branch_paths(
            ref_branch_targets,
            ref_deps,
            gen_deps,
            gen_order
        )
        results['branch_path_accuracy'] = branch_paths_result['average_accuracy']
        results['branch_path_details'] = branch_paths_result
        
        # ========== 指标4: 整体路径正确性 ==========
        # 综合考虑：依赖满足度 + 完整性
        dependency_result = self._evaluate_dependency_satisfaction(
            ref_deps,
            gen_order
        )
        
        # 检查步骤完整性
        ref_steps_set = set(ref_order)
        gen_steps_set = set(gen_order)
        missing_steps = ref_steps_set - gen_steps_set
        extra_steps = gen_steps_set - ref_steps_set
        
        completeness = len(ref_steps_set & gen_steps_set) / len(ref_steps_set) if ref_steps_set else 1.0
        
        # 整体路径正确性 = 依赖满足度 * 完整性
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
    
    def _evaluate_main_path(self, branch_point_id: str, ref_order: List[str], gen_order: List[str]) -> Dict:
        """
        评测主路径（从起点到分支点的线性序列）
        """
        # 提取参考答案的主路径
        try:
            branch_index = ref_order.index(branch_point_id)
            ref_main_path = ref_order[:branch_index + 1]
        except ValueError:
            return {
                'accuracy': 0.0,
                'error': f'Branch point {branch_point_id} not found in reference order'
            }
        
        # 提取生成答案的主路径（按出现顺序收集相同的步骤）
        gen_main_path = []
        for step_id in gen_order:
            if step_id in ref_main_path:
                gen_main_path.append(step_id)
            if step_id == branch_point_id:
                break
        
        # 检查是否完全匹配
        exact_match = (gen_main_path == ref_main_path)
        
        # 计算LCS相似度
        lcs_similarity = self._calculate_lcs_similarity(gen_main_path, ref_main_path)
        
        # 计算位置偏差
        position_errors = []
        ref_positions = {sid: i for i, sid in enumerate(ref_main_path)}
        
        for i, step_id in enumerate(gen_main_path):
            if step_id in ref_positions:
                error = abs(i - ref_positions[step_id])
                position_errors.append(error)
        
        return {
            'reference_path': ref_main_path,
            'generated_path': gen_main_path,
            'exact_match': exact_match,
            'lcs_similarity': lcs_similarity,
            'accuracy': lcs_similarity,
            'position_errors': {
                'mean': sum(position_errors) / len(position_errors) if position_errors else 0.0,
                'max': max(position_errors) if position_errors else 0,
                'total': sum(position_errors)
            }
        }
    
    def _evaluate_branch_structure(self, branch_point_id: str, ref_branch_targets: List[str], gen_deps: Dict[str, List[str]]) -> Dict:
        """
        评测分支结构是否正确
        检查生成的工作流在分支点是否有正确的分支目标
        """
        gen_branch_targets = gen_deps.get(branch_point_id, [])
        
        ref_targets_set = set(ref_branch_targets)
        gen_targets_set = set(gen_branch_targets)
        
        # 完全匹配
        exact_match = (ref_targets_set == gen_targets_set)
        
        # 部分匹配
        intersection = ref_targets_set & gen_targets_set
        union = ref_targets_set | gen_targets_set
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        missing_branches = list(ref_targets_set - gen_targets_set)
        extra_branches = list(gen_targets_set - ref_targets_set)
        
        return {
            'correct': exact_match,
            'reference_branches': ref_branch_targets,
            'generated_branches': gen_branch_targets,
            'jaccard_similarity': jaccard_similarity,
            'missing_branches': missing_branches,
            'extra_branches': extra_branches,
            'matched_branches': list(intersection)
        }
    
    def _evaluate_branch_paths(self, ref_branch_targets: List[str], ref_deps: Dict[str, List[str]], gen_deps: Dict[str, List[str]], gen_order: List[str]) -> Dict:
        """
        评测每个分支路径的准确性
        """
        branch_results = {}
        accuracies = []
        
        for branch_start_id in ref_branch_targets:
            # 提取参考分支的完整路径
            ref_branch_path = self._extract_linear_path(branch_start_id, ref_deps)
            
            # 提取生成分支的完整路径
            gen_branch_path = self._extract_linear_path(branch_start_id, gen_deps)
            
            # 如果生成的工作流中没有这个分支起点，标记为缺失
            if branch_start_id not in gen_deps and branch_start_id not in [s for s in gen_order]:
                branch_results[branch_start_id] = {
                    'reference_path': ref_branch_path,
                    'generated_path': [],
                    'accuracy': 0.0,
                    'status': 'missing'
                }
                accuracies.append(0.0)
                continue
            
            # 检查完全匹配
            exact_match = (gen_branch_path == ref_branch_path)
            
            # 计算LCS相似度
            lcs_similarity = self._calculate_lcs_similarity(gen_branch_path, ref_branch_path)
            
            # 计算位置偏差（在整个工作流中的相对位置）
            ref_positions = {sid: i for i, sid in enumerate(ref_branch_path)}
            gen_positions = {sid: i for i, sid in enumerate(gen_branch_path)}
            
            position_errors = []
            for step_id in ref_branch_path:
                if step_id in gen_positions:
                    error = abs(gen_positions[step_id] - ref_positions[step_id])
                    position_errors.append(error)
            
            branch_results[branch_start_id] = {
                'reference_path': ref_branch_path,
                'generated_path': gen_branch_path,
                'exact_match': exact_match,
                'lcs_similarity': lcs_similarity,
                'accuracy': lcs_similarity,
                'position_errors': {
                    'mean': sum(position_errors) / len(position_errors) if position_errors else 0.0,
                    'max': max(position_errors) if position_errors else 0,
                },
                'status': 'evaluated'
            }
            
            accuracies.append(lcs_similarity)
        
        average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        return {
            'branches': branch_results,
            'average_accuracy': average_accuracy,
            'total_branches': len(ref_branch_targets),
            'evaluated_branches': len(accuracies)
        }
    
    def _extract_linear_path(self, start_step: str, dependencies: Dict[str, List[str]]) -> List[str]:
        """
        从某个步骤开始，提取到终点的完整线性路径
        如果遇到分支点，只取第一个后继（假设每个分支内部是线性的）
        """
        path = [start_step]
        current = start_step
        visited = {start_step}  # 防止循环
        
        while current in dependencies and dependencies[current]:
            next_steps = dependencies[current]
            
            if not next_steps:
                break
            
            # 取第一个后继步骤
            next_step = next_steps[0]
            
            # 防止循环
            if next_step in visited:
                break
            
            path.append(next_step)
            visited.add(next_step)
            current = next_step
        
        return path
    
    def _evaluate_dependency_satisfaction(self, ref_deps: Dict[str, List[str]], gen_order: List[str]) -> Dict:
        """
        评测依赖关系的满足度
        检查生成的顺序是否满足参考答案中的所有依赖关系
        """
        gen_positions = {step_id: i for i, step_id in enumerate(gen_order)}
        
        violations = []
        satisfied_deps = 0
        total_deps = 0
        
        for step_id, next_steps in ref_deps.items():
            for next_id in next_steps:
                total_deps += 1
                
                if step_id not in gen_positions or next_id not in gen_positions:
                    # 如果步骤缺失，记录为违规
                    violations.append({
                        'from': step_id,
                        'to': next_id,
                        'reason': f'Missing step: {step_id if step_id not in gen_positions else next_id}'
                    })
                    continue
                
                # 检查依赖关系：step_id 必须在 next_id 之前
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
        """
        评测线性顺序
        对于线性工作流，只返回一个整体准确度
        """
        # 完全匹配检查
        exact_match = (generated_order == ref_order)
        
        # LCS相似度
        lcs_score = self._calculate_lcs_similarity(generated_order, ref_order)
        
        # 位置偏差统计
        position_errors = self._calculate_position_errors(generated_order, ref_order)
        
        # 检查步骤完整性
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
        
        # 确保 seq2 是较短的序列以节省空间
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

    def _calculate_lis_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """计算两个序列的LIS相似度"""
        if not seq1 or not seq2:
            return 0.0
        
        pos_map = {val: i for i, val in enumerate(seq2)}
        mapped_seq = [pos_map[val] for val in seq1 if val in pos_map]
        
        lis = []
        for val in mapped_seq:
            idx = bisect.bisect_left(lis, val)
            if idx == len(lis):
                lis.append(val)
            else:
                lis[idx] = val
        
        lis_length = len(lis)
        return lis_length / max(len(seq1), len(seq2))
    
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
        
        # 构建生成工作流的依赖图
        gen_deps = {}
        for step in gen_steps:
            if step.get('next_steps'):
                gen_deps[step['step_id']] = set(step['next_steps'])
        
        # 统计准确性
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
        
        # 检查多余的边
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
    
    # def _evaluate_dataflow(self, gen_steps: List[Dict]) -> Dict:
    #     """评测数据流有效性"""
    #     # 构建位置映射
    #     positions = {s['step_id']: i for i, s in enumerate(gen_steps)}
        
    #     violations = []
    #     valid_refs = 0
    #     total_refs = 0
        
    #     # 检查每个步骤的数据依赖
    #     for step in gen_steps:
    #         step_id = step['step_id']
    #         refs = self._find_step_references(step.get('parameters', {}))
    #         refs.extend(self._find_step_references(step.get('condition', '')))
            
    #         for ref_id in refs:
    #             total_refs += 1
    #             if ref_id in positions:
    #                 # 生产者必须在消费者之前
    #                 if positions[ref_id] < positions[step_id]:
    #                     valid_refs += 1
    #                 else:
    #                     violations.append({
    #                         'consumer': step_id,
    #                         'producer': ref_id,
    #                         'consumer_pos': positions[step_id],
    #                         'producer_pos': positions[ref_id]
    #                     })
        
    #     is_valid = len(violations) == 0
    #     validity_score = valid_refs / total_refs if total_refs > 0 else 1.0
        
    #     return {
    #         'is_valid': is_valid,
    #         'validity_score': validity_score,
    #         'total_references': total_refs,
    #         'valid_references': valid_refs,
    #         'violations': violations,
    #         'score': validity_score
    #     }
    
    def _evaluate_control_flow(self, gen_steps: List[Dict]) -> Dict:
        """评测控制流准确性（else_steps, error_handler）"""
        ref_control = self.reference_answer['control_flow']
        
        if not ref_control:
            return {'accuracy': 1.0, 'score': 1.0, 'details': 'No control flow in reference'}
        
        # 构建生成工作流的控制流
        gen_control = {}
        for step in gen_steps:
            cf = {}
            if step.get('else_steps'):
                cf['else_steps'] = set(step['else_steps'])
            if step.get('error_handler'):
                cf['error_handler'] = step['error_handler']
            if cf:
                gen_control[step['step_id']] = cf
        
        # 统计准确性
        total_controls = 0
        correct_controls = 0
        details = {'else_steps': {}, 'error_handler': {}}
        
        for sid, ref_cf in ref_control.items():
            gen_cf = gen_control.get(sid, {})
            
            # 检查else_steps
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
            
            # 检查error_handler
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

    # def _evaluate_topology(self, gen_steps: List[Dict]) -> Dict:
    #     """评测拓扑有效性"""
    #     positions = {s['step_id']: i for i, s in enumerate(gen_steps)}
        
    #     violations = {
    #         'backward_next_steps': [],
    #         'cycles': []
    #     }
        
    #     # 1. 检查next_steps必须前向
    #     for step in gen_steps:
    #         step_id = step['step_id']
    #         for next_id in step.get('next_steps', []):
    #             if next_id in positions:
    #                 if positions[step_id] >= positions[next_id]:
    #                     violations['backward_next_steps'].append({
    #                         'from': step_id,
    #                         'to': next_id,
    #                         'from_pos': positions[step_id],
    #                         'to_pos': positions[next_id]
    #                     })
        
    #     # 2. 检测循环（通过DFS）
    #     cycles = self._detect_cycles(gen_steps)
    #     violations['cycles'] = cycles
        
    #     is_valid = (
    #         len(violations['backward_next_steps']) == 0 and
    #         len(violations['cycles']) == 0
    #     )
        
    #     # 计算拓扑分数
    #     total_issues = len(violations['backward_next_steps']) + len(violations['cycles'])
    #     topology_score = 1.0 if is_valid else max(0.0, 1.0 - total_issues * 0.1)
        
    #     return {
    #         'is_valid': is_valid,
    #         'violations': violations,
    #         'score': topology_score
    #     }
    
    # def _detect_cycles(self, steps: List[Dict]) -> List[List[str]]:
    #     """使用Tarjan算法检测强连通分量（循环）"""
    #     # 构建邻接表
    #     graph = {}
    #     for s in steps:
    #         sid = s['step_id']
    #         graph[sid] = []
    #         graph[sid].extend(s.get('next_steps', []))
    #         graph[sid].extend(s.get('else_steps', []))
    #         if s.get('error_handler'):
    #             graph[sid].append(s['error_handler'])
        
    #     # Tarjan算法
    #     index_counter = [0]
    #     stack = []
    #     lowlinks = {}
    #     index = {}
    #     on_stack = set()
    #     sccs = []
        
    #     def strongconnect(node):
    #         index[node] = index_counter[0]
    #         lowlinks[node] = index_counter[0]
    #         index_counter[0] += 1
    #         stack.append(node)
    #         on_stack.add(node)
            
    #         for successor in graph.get(node, []):
    #             if successor not in index:
    #                 strongconnect(successor)
    #                 lowlinks[node] = min(lowlinks[node], lowlinks[successor])
    #             elif successor in on_stack:
    #                 lowlinks[node] = min(lowlinks[node], index[successor])
            
    #         if lowlinks[node] == index[node]:
    #             scc = []
    #             while True:
    #                 successor = stack.pop()
    #                 on_stack.remove(successor)
    #                 scc.append(successor)
    #                 if successor == node:
    #                     break
    #             if len(scc) > 1:  # 只记录真正的循环
    #                 sccs.append(scc)
        
    #     for node in graph:
    #         if node not in index:
    #             strongconnect(node)
        
    #     return sccs
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """
        计算总分
        """
        order_metrics = metrics.get('order_accuracy', {})
        
        if order_metrics.get('evaluation_type') == 'branching_workflow':
            # 对于分支工作流，使用4个独立指标的加权平均
            weights = {
                'overall_path_correctness': 0.30,
                'main_path_accuracy': 0.25,
                'branch_structure_correctness': 0.20,
                'branch_path_accuracy': 0.25
            }
            
            order_score = sum(
                order_metrics.get(key, 0.0) * weight 
                for key, weight in weights.items()
            )
        else:
            # 线性工作流
            order_score = order_metrics.get('overall_path_correctness', 0.0)
        
        # 其他指标
        weights = {
            'order': 0.50,  # 执行顺序占50%
            'dependency_accuracy': 0.25,
            # 'dataflow_validity': 0.15,
            'control_flow_accuracy': 0.25,
            # 'topology_validity': 0.075
        }
        
        total_score = (
            weights['order'] * order_score +
            weights['dependency_accuracy'] * metrics.get('dependency_accuracy', {}).get('score', 0.0) +
            # weights['dataflow_validity'] * metrics.get('dataflow_validity', {}).get('score', 0.0) +
            weights['control_flow_accuracy'] * metrics.get('control_flow_accuracy', {}).get('score', 0.0)
            # + weights['topology_validity'] * metrics.get('topology_validity', {}).get('score', 0.0)
        )
        
        return total_score
    
    def generate_report(self, results: Dict, output_path: Path = None) -> str:
        """生成详细评测报告"""
        lines = [
            "=" * 80,
            "Workflow Orchestration Evaluation Report",
            "=" * 80,
            f"Timestamp: {results['timestamp']}",
            "",
        ]
        
        # 执行顺序评测结果
        order_metrics = results['metrics']['order_accuracy']
        
        if order_metrics.get('evaluation_type') == 'branching_workflow':
            lines.extend([
                "=" * 80,
                "EXECUTION ORDER EVALUATION (BRANCHING WORKFLOW)",
                "=" * 80,
                "",
            ])
            
            # ========== 4个独立指标 ==========
            lines.extend([
                "📊 KEY METRICS:",
                "",
                f"1. Overall Path Correctness:      {order_metrics['overall_path_correctness']:.2%}",
                f"2. Main Path Accuracy:            {order_metrics['main_path_accuracy']:.2%}",
                f"3. Branch Structure Correctness:  {order_metrics['branch_structure_correctness']:.2%}",
                f"4. Branch Path Accuracy:          {order_metrics['branch_path_accuracy']:.2%}",
                "",
            ])
            
            # ========== 详细分析 ==========
            lines.extend([
                "=" * 80,
                "DETAILED ANALYSIS",
                "=" * 80,
                "",
            ])
            
            # 1. 整体路径正确性详情
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
            
            # 2. 主路径准确度详情
            main_path = order_metrics['main_path_details']
            lines.extend([
                "2️⃣  MAIN PATH ACCURACY",
                f"   - Exact Match: {'✓' if main_path.get('exact_match', False) else '✗'}",
                f"   - LCS Similarity: {main_path.get('lcs_similarity', 0):.2%}",
                f"   - Reference Path:",
                f"     {' -> '.join(main_path.get('reference_path', []))}",
                f"   - Generated Path:",
                f"     {' -> '.join(main_path.get('generated_path', []))}",
            ])
            
            pos_errors = main_path.get('position_errors', {})
            if pos_errors.get('total', 0) > 0:
                lines.extend([
                    f"   - Position Errors:",
                    f"     • Mean: {pos_errors.get('mean', 0):.2f}",
                    f"     • Max: {pos_errors.get('max', 0)}",
                ])
            
            lines.append("")
            
            # 3. 分支结构正确性详情
            branch_struct = order_metrics['branch_structure_details']
            lines.extend([
                "3️⃣  BRANCH STRUCTURE CORRECTNESS",
                f"   - Exact Match: {'✓' if branch_struct.get('correct', False) else '✗'}",
                f"   - Jaccard Similarity: {branch_struct.get('jaccard_similarity', 0):.2%}",
                f"   - Reference Branches: {branch_struct.get('reference_branches', [])}",
                f"   - Generated Branches: {branch_struct.get('generated_branches', [])}",
            ])
            
            if branch_struct.get('matched_branches'):
                lines.append(f"   - Matched Branches: {branch_struct['matched_branches']}")
            if branch_struct.get('missing_branches'):
                lines.append(f"   - Missing Branches: {branch_struct['missing_branches']}")
            if branch_struct.get('extra_branches'):
                lines.append(f"   - Extra Branches: {branch_struct['extra_branches']}")
            
            lines.append("")
            
            # 4. 分支路径准确度详情
            branch_paths = order_metrics['branch_path_details']
            lines.extend([
                "4️⃣  BRANCH PATH ACCURACY",
                f"   - Average Accuracy: {branch_paths.get('average_accuracy', 0):.2%}",
                f"   - Evaluated Branches: {branch_paths.get('evaluated_branches', 0)}/{branch_paths.get('total_branches', 0)}",
                "",
            ])
            
            for branch_id, branch_info in branch_paths.get('branches', {}).items():
                status_icon = "✓" if branch_info.get('exact_match', False) else "✗"
                lines.extend([
                    f"   Branch: {branch_id} {status_icon}",
                    f"   - Accuracy: {branch_info.get('accuracy', 0):.2%}",
                    f"   - Reference: {' -> '.join(branch_info.get('reference_path', []))}",
                    f"   - Generated: {' -> '.join(branch_info.get('generated_path', []))}",
                ])
                
                if branch_info.get('status') == 'missing':
                    lines.append(f"   - Status: MISSING")
                
                lines.append("")
        
        else:
            # 线性工作流
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
        
        # 其他指标（2-5）
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
        
        # df_metrics = results['metrics']['dataflow_validity']
        # lines.extend([
        #     "Dataflow Validity",
        #     f"- Valid: {'✓' if df_metrics['is_valid'] else '✗'}",
        #     f"- Validity Score: {df_metrics['validity_score']:.2%}",
        #     f"- Valid References: {df_metrics['valid_references']}/{df_metrics['total_references']}",
        #     "",
        # ])
        
        cf_metrics = results['metrics']['control_flow_accuracy']
        lines.extend([
            "Control Flow Accuracy",
            f"- Accuracy: {cf_metrics['accuracy']:.2%}",
            "",
        ])
        
        # topo_metrics = results['metrics']['topology_validity']
        # lines.extend([
        #     "Topology Validity",
        #     f"- Valid: {'✓' if topo_metrics['is_valid'] else '✗'}",
        #     f"- Score: {topo_metrics['score']:.2%}",
        #     "",
        # ])
        
        # 可视化对比
        lines.extend([
            "=" * 80,
            "WORKFLOW STRUCTURE COMPARISON",
            "=" * 80,
            "",
        ])
        
        # 构建生成工作流的依赖图
        gen_deps = {}
        if hasattr(self, 'generated_workflow_steps'):
            for step in self.generated_workflow_steps:
                if step.get('next_steps'):
                    gen_deps[step['step_id']] = step['next_steps']
        
        # 显示参考结构
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
        
        # 显示生成结构
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
        
        # 最终判定
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



# parse_llm_output 已移动到 `utils.evaluate_tools.parse_llm_output`


import asyncio
import json
import os
from pathlib import Path


async def main():
    print("=" * 80)
    print("Unified Workflow Orchestration Evaluation System - Batch Mode")
    print("=" * 80)

    # 配置LLM
    llm_config = LLMConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        api_key="sk-2p51ZI79J5X4OL6S343c17F08f3c432395C711608b2eB0D5",
        base_url="https://az.gptplus5.com/v1",
        temperature=0.7,
        max_tokens=2048,
        timeout=60.0,
        retry_attempts=3,
        retry_delay=2.0
    )
    llm_interface = LLMInterface(llm_config)

    # 根目录路径
    reference_workflow_root_path = Path('E:\\MCPBenchMark\\MCPFLow\\mcp_projects')
    
    # 输出根目录
    output_root_dir = Path('./evaluation_output')
    output_root_dir.mkdir(exist_ok=True)
    
    # 收集所有 mcp_project 目录
    mcp_projects = []
    for item in reference_workflow_root_path.iterdir():
        if item.is_dir() and item.name.startswith('mcp_project'):
            # 提取数字序号
            try:
                project_num = int(item.name.replace('mcp_project', ''))
                mcp_projects.append((project_num, item))
            except ValueError:
                print(f"⚠️ Skipping invalid project name: {item.name}")
                continue
    
    # 按序号排序
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
        print(f"\n{'='*60}")
        print(f"Processing mcp_project{project_num}")
        print(f"{'='*60}")
        
        # 创建该项目的输出目录（用数字序号命名）
        project_output_dir = output_root_dir / str(project_num)
        project_output_dir.mkdir(exist_ok=True)
        
        workflow_path = project_path / 'workflow.json'
        
        # 检查 workflow.json 是否存在
        if not workflow_path.exists():
            skip_msg = f"workflow.json not found in mcp_project{project_num}"
            print(f"⚠️ {skip_msg}")
            
            # 记录跳过信息
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
        
        try:
            # 初始化评测系统（启用混淆）
            eval_system = WorkflowEvaluationSystem(
                reference_workflow_path=str(workflow_path),
                seed=42,
                obfuscate=True
            )
            
            # 生成打乱的测试工作流
            shuffled_workflow = eval_system.generate_shuffled_workflow()
            
            # 保存测试用例
            with open(project_output_dir / 'shuffled_workflow.json', 'w', encoding='utf-8') as f:
                json.dump(shuffled_workflow, f, indent=2, ensure_ascii=False)
            
            with open(project_output_dir / 'reference_answer.json', 'w', encoding='utf-8') as f:
                json.dump(eval_system.reference_answer, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Generated shuffled workflow (Obfuscated: {eval_system.obfuscate})")
            print(f"   Steps: {len(shuffled_workflow['workflow_steps'])}")
            
            # 根据是否有分支显示不同的信息
            if eval_system.reference_answer.get('has_branches'):
                print(f"   Workflow Type: BRANCHING (DAG)")
                print(f"   Main Path: {' -> '.join(eval_system.reference_answer['execution_order'][:5])}")
                if len(eval_system.reference_answer['execution_order']) > 4:
                    print(f"   Branch Point: Step 5 ({eval_system.reference_answer['execution_order'][4]})")
                    
                    deps = eval_system.reference_answer['dependencies']
                    branch_point = eval_system.reference_answer['execution_order'][4]
                    if branch_point in deps and len(deps[branch_point]) > 1:
                        print(f"   Branches:")
                        for i, branch_id in enumerate(deps[branch_point], 1):
                            branch_path = [branch_id]
                            current = branch_id
                            while current in deps and deps[current]:
                                next_id = deps[current][0]
                                branch_path.append(next_id)
                                current = next_id
                            print(f"     Branch {i}: {' -> '.join(branch_path)}")
            else:
                print(f"   Workflow Type: LINEAR")
                print(f"   Reference Order: {' -> '.join(eval_system.reference_answer['execution_order'])}")
            
            if eval_system.id_mapping:
                print(f"   ID Obfuscation: {len(eval_system.id_mapping)} steps")
            
            # 构建提示词
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
            {json.dumps(shuffled_workflow, indent=2)}

            Return the workflow_steps array in the correct execution order."""

            print("\n📤 Sending to LLM...")
            print(f"   Prompt length: {len(prompt)} characters")
            
            try:
                llm_output = await llm_interface.generate_response(prompt)
                # print(f"✅ Received response ({len(llm_output)} chars)")
                with open(project_output_dir / 'llm_response.txt', 'w', encoding='utf-8') as f:
                    f.write(llm_output)
            except asyncio.TimeoutError:
                raise Exception("LLM request timed out")
            except Exception as e:
                raise Exception(f"LLM request failed: {type(e).__name__}: {e}")
            
            # 保存LLM原始输出
            with open(project_output_dir / 'llm_raw_output.txt', 'w', encoding='utf-8') as f:
                f.write(llm_output)
            
            # 解析LLM输出
            try:
                generated_workflow = parse_llm_output(llm_output)
                print("✅ Successfully parsed LLM output")
                
                with open(project_output_dir / 'generated_workflow.json', 'w', encoding='utf-8') as f:
                    json.dump(generated_workflow, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                raise Exception(f"Failed to parse LLM output: {e}")
            
            # 评测
            print("\n🔍 Evaluating...")
            results = eval_system.evaluate(generated_workflow)
            
            # 生成报告
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
                'results': results
            })
            
            print(f"\n✅ mcp_project{project_num} evaluation complete!")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error processing mcp_project{project_num}: {error_msg}")
            
            # 记录错误信息
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
    print(f"Skipped (no workflow.json): {summary_results['skipped']}")
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
        "BATCH EVALUATION SUMMARY REPORT",
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
            score = results.get('final_score', 'N/A')
            lines.append(f"Project {project_num}: ✅ SUCCESS (Score: {score})")
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
