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
import asyncio
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from utils.llm_interface import LLMInterface
from core.config import LLMConfig
from utils.evaluate_tools import obfuscate_steps, parse_llm_output
from utils.mcp_extractor import MCPToolInfo, MCPServerToolsExtractor
from utils.workflow_helpers import (
    calculate_lcs_similarity, calculate_position_errors, find_step_references,
    extract_linear_path, evaluate_dependency_satisfaction, generate_summary_report,
    check_has_branches, extract_dependencies, extract_dataflow, extract_control_flow
)
from utils.report_generator import generate_evaluation_report
from utils.error_propagation import calculate_error_propagation

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
            'dependencies': extract_dependencies(steps),
            'dataflow': extract_dataflow(steps),
            'control_flow': extract_control_flow(steps),
            'step_count': len(steps),
            'has_branches': check_has_branches(steps)
        }
    
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

        # 4. 错误传播率
        results['metrics']['error_propagation'] = self._evaluate_error_propagation(gen_steps)
        
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
        
        dependency_result = evaluate_dependency_satisfaction(ref_deps, gen_order)
        
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

    def _evaluate_linear_order(self, generated_order: List[str], ref_order: List[str]) -> Dict:
        """评测线性顺序"""
        exact_match = (generated_order == ref_order)
        lcs_score = calculate_lcs_similarity(generated_order, ref_order)
        position_errors = calculate_position_errors(generated_order, ref_order)
        
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
    
    def _evaluate_error_propagation(self, gen_steps: List[Dict]) -> Dict:
        '''评测错误传播率'''
        
        return calculate_error_propagation(
            ref_order=self.reference_answer['execution_order'],
            gen_order=[s['step_id'] for s in gen_steps],
            ref_deps=self.reference_answer['dependencies'],
            gen_steps=gen_steps
        )
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """计算总分"""
        order_metrics = metrics.get('order_accuracy', {})
        order_score = order_metrics.get('overall_path_correctness', 0.0)
        
        weights = {
            'order': 0.40,                    
            'dependency_accuracy': 0.20,      
            'control_flow_accuracy': 0.20,    
            'error_propagation': 0.20,        
        }
        
        total_score = (
            weights['order'] * order_score +
            weights['dependency_accuracy'] * metrics.get('dependency_accuracy', {}).get('score', 0.0) +
            weights['control_flow_accuracy'] * metrics.get('control_flow_accuracy', {}).get('score', 0.0) +
            weights['error_propagation'] * metrics.get('error_propagation', {}).get('score', 0.0)
        )
        
        return total_score
    
    def generate_report(self, results: Dict, output_path: Path = None) -> str:
        """生成详细评测报告"""
        return generate_evaluation_report(
            results=results,
            mcp_tools=self.mcp_tools,
            reference_answer=self.reference_answer,
            generated_workflow_steps=getattr(self, 'generated_workflow_steps', []),
            output_path=output_path
        )

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
    reference_workflow_root_path = Path('D:\\MCPFLow\\mcp_projects')
    
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

        # if project_num != 23: 
        #     continue 

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

if __name__ == '__main__':
    asyncio.run(main())
