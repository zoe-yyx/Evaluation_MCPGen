"""
tasks/mcp_tool_evaluator.py 
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

from core.config import EvaluationConfig
from utils.llm_interface import LLMInterface
from tasks.mcp_tool_regenerator import MCPToolRegenerator, RegenerationResult

logger = logging.getLogger(__name__)


@dataclass
class DatasetSummary:
    """单个数据集的统计"""
    dataset_name: str
    
    total_workflows: int = 0
    total_tools: int = 0
    
    # 静态分析
    static_passed: int = 0
    
    # 单工具测试（Individual）
    individual_passed: int = 0
    
    # 整体测试（Collective）
    collective_passed: int = 0  # 整体测试通过的工作流数
    
    # 高质量工具
    high_quality_tools: int = 0
    
    total_time: float = 0.0
    
    workflow_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_rates(self) -> Dict[str, float]:
        """计算各种通过率"""
        return {
            'static_pass_rate': self.static_passed / self.total_tools if self.total_tools > 0 else 0.0,
            'individual_pass_rate': self.individual_passed / self.total_tools if self.total_tools > 0 else 0.0,
            'collective_pass_rate': self.collective_passed / self.total_workflows if self.total_workflows > 0 else 0.0,
            'high_quality_rate': self.high_quality_tools / self.total_tools if self.total_tools > 0 else 0.0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        rates = self.get_rates()
        return {
            'dataset_name': self.dataset_name,
            'statistics': {
                'workflows': {
                    'total': self.total_workflows,
                    'collective_passed': self.collective_passed,
                    'collective_pass_rate': rates['collective_pass_rate'],
                },
                'tools': {
                    'total': self.total_tools,
                    'static_passed': self.static_passed,
                    'static_pass_rate': rates['static_pass_rate'],
                    'individual_passed': self.individual_passed,
                    'individual_pass_rate': rates['individual_pass_rate'],
                    'high_quality': self.high_quality_tools,
                    'high_quality_rate': rates['high_quality_rate'],
                },
                'time': self.total_time,
            },
            'workflow_results': self.workflow_results,
        }


@dataclass
class EvaluationSummary:
    """总体评测结果"""
    
    total_datasets: int = 0
    total_workflows: int = 0
    total_tools: int = 0
    
    # 各阶段通过数
    static_passed: int = 0
    individual_passed: int = 0
    collective_passed: int = 0
    high_quality_tools: int = 0
    
    total_time: float = 0.0
    
    dataset_summaries: List[DatasetSummary] = field(default_factory=list)
    
    def get_rates(self) -> Dict[str, float]:
        """计算各种通过率"""
        return {
            'static_pass_rate': self.static_passed / self.total_tools if self.total_tools > 0 else 0.0,
            'individual_pass_rate': self.individual_passed / self.total_tools if self.total_tools > 0 else 0.0,
            'collective_pass_rate': self.collective_passed / self.total_workflows if self.total_workflows > 0 else 0.0,
            'high_quality_rate': self.high_quality_tools / self.total_tools if self.total_tools > 0 else 0.0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        rates = self.get_rates()
        return {
            'overall_statistics': {
                'datasets': self.total_datasets,
                'workflows': {
                    'total': self.total_workflows,
                    'collective_passed': self.collective_passed,
                    'collective_pass_rate': rates['collective_pass_rate'],
                },
                'tools': {
                    'total': self.total_tools,
                    'static_passed': self.static_passed,
                    'static_pass_rate': rates['static_pass_rate'],
                    'individual_passed': self.individual_passed,
                    'individual_pass_rate': rates['individual_pass_rate'],
                    'high_quality': self.high_quality_tools,
                    'high_quality_rate': rates['high_quality_rate'],
                },
                'time': self.total_time,
            },
            'dataset_summaries': [ds.to_dict() for ds in self.dataset_summaries],
        }


class MCPToolEvaluator:
    """MCP 工具评测器"""
    
    def __init__(self, config: EvaluationConfig, llm_interface: LLMInterface):
        self.config = config
        self.llm = llm_interface
        self.regenerator = MCPToolRegenerator(config, llm_interface)
    
    async def evaluate_dataset(
        self,
        dataset_path: Path,
        output_dir: Optional[Path] = None
    ) -> EvaluationSummary:
        """
        评测数据集
        
        对每个工作流同时执行两种测试：
        1. Individual Test: 每个工具单独替换测试
        2. Collective Test: 所有工具一起替换测试
        """
        start_time = time.time()
        
        # 创建输出目录
        if output_dir is None:
            output_dir = Path("output")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        
        # 备份配置
        self._backup_config(output_dir)
        
        # 查找数据集目录
        dataset_dirs = self._find_dataset_directories(dataset_path)
        
        if not dataset_dirs:
            logger.error(f"No datasets found in: {dataset_path}")
            return EvaluationSummary()
        
        logger.info(f"Found {len(dataset_dirs)} dataset(s) to evaluate")
        
        # 初始化总结
        summary = EvaluationSummary()
        summary.total_datasets = len(dataset_dirs)
        
        # 评测每个数据集
        for i, dataset_dir in enumerate(dataset_dirs, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating dataset {i}/{len(dataset_dirs)}: {dataset_dir.name}")
            logger.info(f"{'='*80}")
            
            dataset_summary = await self._evaluate_single_dataset(
                dataset_dir=dataset_dir,
                output_dir=output_dir / dataset_dir.name
            )
            
            # 汇总统计
            summary.dataset_summaries.append(dataset_summary)
            summary.total_workflows += dataset_summary.total_workflows
            summary.total_tools += dataset_summary.total_tools
            summary.static_passed += dataset_summary.static_passed
            summary.individual_passed += dataset_summary.individual_passed
            summary.collective_passed += dataset_summary.collective_passed
            summary.high_quality_tools += dataset_summary.high_quality_tools
        
        summary.total_time = time.time() - start_time
        
        # 保存总结
        self._save_total_summary(summary, output_dir)
        
        # 打印总结
        self._print_summary(summary)
        
        return summary
    
    async def _evaluate_single_dataset(
        self,
        dataset_dir: Path,
        output_dir: Path
    ) -> DatasetSummary:
        """评测单个数据集"""
        start_time = time.time()
        
        summary = DatasetSummary(dataset_name=dataset_dir.name)
        
        # 查找工作流目录
        workflow_dirs = self._find_workflow_directories(dataset_dir)
        summary.total_workflows = len(workflow_dirs)
        
        if summary.total_workflows == 0:
            logger.warning(f"No workflows found in dataset: {dataset_dir.name}")
            return summary
        
        logger.info(f"Found {summary.total_workflows} workflow(s)")
        
        # 评测每个工作流
        for workflow_dir in workflow_dirs:
            logger.info(f"\nProcessing workflow: {workflow_dir.name}")
            
            try:
                # 运行评测（同时执行两种测试）
                result: RegenerationResult = await self.regenerator.run_regeneration_evaluation(
                    workflow_dir=workflow_dir,
                    output_dir=output_dir / workflow_dir.name
                )
                
                # 统计
                summary.workflow_results.append({
                    'workflow_name': result.workflow_name,
                    'total_tools': result.total_tools,
                    'static_passed': result.static_analysis_passed,
                    'individual_passed': result.individual_test_passed,
                    'collective_success': result.collective_test_success,
                    'high_quality': result.high_quality_tools,
                })
                
                summary.total_tools += result.total_tools
                summary.static_passed += result.static_analysis_passed
                summary.individual_passed += result.individual_test_passed
                summary.high_quality_tools += result.high_quality_tools
                
                if result.collective_test_success:
                    summary.collective_passed += 1
                
            except Exception as e:
                logger.error(f"Failed to evaluate workflow {workflow_dir.name}: {e}")
                summary.workflow_results.append({
                    'workflow_name': workflow_dir.name,
                    'error': str(e),
                })
        
        summary.total_time = time.time() - start_time

        self._save_dataset_summary(summary, output_dir)
        
        return summary
    
    def _find_dataset_directories(self, dataset_path: Path) -> List[Path]:
        """查找数据集目录"""
        if not dataset_path.exists():
            return []

        if (dataset_path / "workflow.json").exists():
            return [dataset_path]
        
        dataset_dirs = []
        for subdir in dataset_path.iterdir():
            if not subdir.is_dir():
                continue
            
            if (subdir / "workflow.json").exists():
                dataset_dirs.append(subdir)
            else:
                has_workflows = any(
                    (nested / "workflow.json").exists()
                    for nested in subdir.iterdir()
                    if nested.is_dir()
                )
                if has_workflows:
                    dataset_dirs.append(subdir)
        
        return sorted(dataset_dirs)
    
    def _find_workflow_directories(self, dataset_dir: Path) -> List[Path]:
        """查找工作流目录"""
        if (dataset_dir / "workflow.json").exists():
            return [dataset_dir]
        
        workflows = []
        for subdir in dataset_dir.iterdir():
            if subdir.is_dir() and (subdir / "workflow.json").exists():
                workflows.append(subdir)
        
        return sorted(workflows)
    
    def _backup_config(self, output_dir: Path):
        """备份配置"""
        import datetime
        
        config_backup = {
            'backup_time': datetime.datetime.now().isoformat(),
            'config': self.config.to_dict(),
        }
        
        with open(output_dir / "config_backup.json", 'w') as f:
            json.dump(config_backup, f, indent=2)
    
    def _save_dataset_summary(self, summary: DatasetSummary, output_dir: Path):
        """保存数据集总结"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_total_summary(self, summary: EvaluationSummary, output_dir: Path):
        """保存总体总结"""
        with open(output_dir / "total_summary.json", 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _print_summary(self, summary: EvaluationSummary):
        """打印总结"""
        rates = summary.get_rates()
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\n📁 Datasets: {summary.total_datasets}")
        print(f"📋 Workflows: {summary.total_workflows}")
        print(f"🔧 Tools: {summary.total_tools}")
        
        print(f"\n📊 Results:")
        print(f"   Static Analysis Passed:  {summary.static_passed}/{summary.total_tools} "
              f"({rates['static_pass_rate']*100:.1f}%)")
        print(f"   Individual Test Passed:  {summary.individual_passed}/{summary.total_tools} "
              f"({rates['individual_pass_rate']*100:.1f}%)")
        print(f"   Collective Test Passed:  {summary.collective_passed}/{summary.total_workflows} workflows "
              f"({rates['collective_pass_rate']*100:.1f}%)")
        print(f"   High Quality Tools:      {summary.high_quality_tools}/{summary.total_tools} "
              f"({rates['high_quality_rate']*100:.1f}%)")
        
        print(f"\n⏱️  Total Time: {summary.total_time:.2f}s")
        
        print("\n" + "=" * 80)


async def main():
    import argparse
    from utils.logger import setup_logger
    
    parser = argparse.ArgumentParser(description="MCP Tool Evaluator")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    setup_logger()
    
    config = EvaluationConfig(args.config)
    llm = LLMInterface(config.llm_config)
    
    evaluator = MCPToolEvaluator(config, llm)
    
    await evaluator.evaluate_dataset(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output)
    )

if __name__ == "__main__":
    asyncio.run(main())