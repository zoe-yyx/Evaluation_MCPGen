"""
Test script for simplified MCP tool evaluation system

This script tests the simplified MCP tool regeneration and evaluation functionality.
The evaluation focuses on two key metrics:
1. Syntax correctness of generated tools
2. Workflow execution success with regenerated tools
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import EvaluationConfig
from tasks.mcp_tool_evaluator import MCPToolEvaluator
from utils.llm_interface import LLMInterface
from utils.logger import setup_logger

logger = setup_logger("test_mcp_tool_evaluation")


async def test_single_workflow():
    """Test simplified regeneration and evaluation for a single workflow"""
    # Load configuration
    config = EvaluationConfig("config.yaml")

    # Initialize LLM interface
    llm = LLMInterface(config.llm_config)

    # Initialize evaluator
    evaluator = MCPToolEvaluator(config, llm)

    # Get dataset path from config
    dataset_path = Path(config.dataset_path)

    # Run evaluation
    logger.info(f"Testing simplified evaluation for single workflow: {dataset_path}")
    logger.info("Evaluation focuses on:")
    logger.info("1. Syntax correctness of generated tools")
    logger.info("2. Workflow execution success with regenerated tools")

    output_dir = Path(config.output_path) if config.output_path else None
    summary = await evaluator.evaluate_dataset(dataset_path, output_dir)

    # Log detailed results
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"Total workflows: {summary.total_workflows}")
    logger.info(f"Successful workflows: {summary.successful_workflows}")
    logger.info(f"Total tools: {summary.total_tools}")
    logger.info(f"Tools with correct syntax: {summary.syntax_correct_tools}")
    logger.info(f"Tools that passed workflow tests: {summary.successful_tools}")
    logger.info(f"Workflow success rate: {summary.execution_success_rate:.2%}")
    logger.info(f"Syntax correctness rate: {summary.syntax_success_rate:.2%}")
    logger.info(f"Average execution time: {summary.average_execution_time:.2f}s")

    # Print individual dataset and workflow results
    for dataset_summary in summary.dataset_summaries:
        logger.info(f"\n--- Dataset: {dataset_summary.dataset_name} ---")
        logger.info(f"  Dataset workflows: {dataset_summary.total_workflows}")
        logger.info(
            f"  Dataset success rate: {dataset_summary.execution_success_rate:.2%}"
        )

        for result in dataset_summary.results:
            logger.info(f"\n  --- Workflow: {result.workflow_name} ---")
            logger.info(f"    Overall workflow success: {result.execution_success}")
            logger.info(f"    Error message: {result.error_message or 'None'}")
            logger.info(f"    Tools regenerated: {len(result.tools)}")

            for tool in result.tools:
                status_indicators = []
                if tool.syntax_correct:
                    status_indicators.append("✓ Syntax OK")
                else:
                    status_indicators.append("✗ Syntax Error")

                if tool.execution_success:
                    status_indicators.append("✓ Workflow OK")
                else:
                    status_indicators.append("✗ Workflow Failed")

                logger.info(f"      - {tool.name}: {' | '.join(status_indicators)}")

    # Summary of key findings
    logger.info("\n=== Key Findings ===")
    syntax_rate = summary.syntax_success_rate
    workflow_rate = summary.execution_success_rate

    if syntax_rate >= 0.8:
        logger.info(f"✓ High syntax correctness rate: {syntax_rate:.1%}")
    elif syntax_rate >= 0.5:
        logger.info(f"⚠ Moderate syntax correctness rate: {syntax_rate:.1%}")
    else:
        logger.info(f"✗ Low syntax correctness rate: {syntax_rate:.1%}")

    if workflow_rate >= 0.8:
        logger.info(f"✓ High workflow success rate: {workflow_rate:.1%}")
    elif workflow_rate >= 0.5:
        logger.info(f"⚠ Moderate workflow success rate: {workflow_rate:.1%}")
    else:
        logger.info(f"✗ Low workflow success rate: {workflow_rate:.1%}")

    return summary


async def test_full_dataset():
    """Test simplified regeneration and evaluation for the full dataset"""
    # Load configuration
    config = EvaluationConfig("config.yaml")

    # Initialize LLM interface
    llm = LLMInterface(config.llm_config)

    # Initialize evaluator
    evaluator = MCPToolEvaluator(config, llm)

    # Get dataset path from config - for full dataset, use parent directory
    config_dataset_path = Path(config.dataset_path)
    if config_dataset_path.name != "dataset":
        # If config points to a specific project, use its parent for full dataset test
        dataset_path = config_dataset_path.parent
    else:
        dataset_path = config_dataset_path

    # Run evaluation
    logger.info(f"Testing simplified evaluation for full dataset: {dataset_path}")
    logger.info("Evaluation focuses on:")
    logger.info("1. Syntax correctness of generated tools")
    logger.info("2. Workflow execution success with regenerated tools")

    output_dir = Path(config.output_path) if config.output_path else None
    summary = await evaluator.evaluate_dataset(dataset_path, output_dir)

    # Log summary results
    logger.info("\n=== Dataset Evaluation Summary ===")
    logger.info(f"Total datasets processed: {summary.total_datasets}")
    logger.info(f"Total workflows processed: {summary.total_workflows}")
    logger.info(f"Workflows with successful execution: {summary.successful_workflows}")
    logger.info(f"Total tools regenerated: {summary.total_tools}")
    logger.info(f"Tools with correct syntax: {summary.syntax_correct_tools}")
    logger.info(f"Tools that passed workflow tests: {summary.successful_tools}")
    logger.info(f"Overall workflow success rate: {summary.execution_success_rate:.2%}")
    logger.info(f"Overall syntax correctness rate: {summary.syntax_success_rate:.2%}")
    logger.info(
        f"Average execution time per workflow: {summary.average_execution_time:.2f}s"
    )

    # Log individual dataset results
    for dataset_summary in summary.dataset_summaries:
        logger.info(f"\n--- Dataset: {dataset_summary.dataset_name} ---")
        logger.info(f"  Workflows: {dataset_summary.total_workflows}")
        logger.info(f"  Success rate: {dataset_summary.execution_success_rate:.2%}")
        logger.info(f"  Syntax rate: {dataset_summary.syntax_success_rate:.2%}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test simplified MCP tool evaluation")
    parser.add_argument(
        "--mode",
        choices=["single", "full"],
        default="single",
        help="Test mode: single workflow or full dataset",
    )

    args = parser.parse_args()

    if args.mode == "single":
        asyncio.run(test_single_workflow())
    else:
        asyncio.run(test_full_dataset())
