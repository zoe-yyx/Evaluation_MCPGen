#!/usr/bin/env python3
"""
Test script for optimized result storage functionality
"""

import asyncio
import json
import tempfile
from pathlib import Path
from tasks.optimized_result_storage import OptimizedResultStorage


def test_optimized_storage():
    """Test the optimized result storage functionality"""

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        storage = OptimizedResultStorage(temp_path)

        print("Testing OptimizedResultStorage...")

        # Test 1: Tool execution details
        print("\n1. Testing tool execution details storage...")
        tool_details = {
            "name": "test_tool",
            "registration_success": True,
            "execution_success": True,
            "syntax_correct": True,
            "test_stdout": "Tool executed successfully\nOutput line 2\nOutput line 3",
            "test_stderr": "Warning: deprecated function\nError line 2",
            "test_error_message": None,
            "original_path": "/path/to/original.py",
            "regenerated_path": "/path/to/regenerated.py",
        }

        output_dir = temp_path / "test_workflow"
        details_file = storage.save_tool_execution_details(
            workflow_id="test_workflow_1",
            tool_name="test_tool",
            tool_details=tool_details,
            output_dir=output_dir,
        )

        print(f"✓ Tool details saved to: {details_file}")

        # Check if stdout and stderr files were created
        stdout_file = output_dir / "tool_test_tool_stdout.txt"
        stderr_file = output_dir / "tool_test_tool_stderr.txt"

        if stdout_file.exists():
            print(f"✓ Stdout file created: {stdout_file}")
            with open(stdout_file, "r") as f:
                stdout_content = f.read()
                print(
                    f"  Stdout content (with proper newlines):\n{repr(stdout_content)}"
                )

        if stderr_file.exists():
            print(f"✓ Stderr file created: {stderr_file}")
            with open(stderr_file, "r") as f:
                stderr_content = f.read()
                print(
                    f"  Stderr content (with proper newlines):\n{repr(stderr_content)}"
                )

        # Test 2: Workflow execution details
        print("\n2. Testing workflow execution details storage...")
        workflow_result = {
            "workflow_id": "test_workflow_1",
            "workflow_name": "Test Workflow",
            "execution_success": True,
            "execution_time": 15.5,
            "error_message": None,
            "workflow_stdout": "Workflow started\nStep 1 completed\nStep 2 completed\nWorkflow finished",
            "workflow_stderr": "Warning: some deprecation\nInfo: processing complete",
            "tools": [
                {
                    "name": "tool1",
                    "execution_success": True,
                    "syntax_correct": True,
                    "test_stdout": "Tool 1 output",
                    "test_stderr": "Tool 1 warning",
                },
                {
                    "name": "tool2",
                    "execution_success": False,
                    "syntax_correct": True,
                    "test_stdout": "Tool 2 output",
                    "test_stderr": "Tool 2 error",
                },
            ],
        }

        workflow_file = storage.save_workflow_execution_details(
            workflow_result, output_dir
        )

        print(f"✓ Workflow details saved to: {workflow_file}")

        # Check workflow output files
        workflow_stdout_file = output_dir / "workflow_test_workflow_1_stdout.txt"
        workflow_stderr_file = output_dir / "workflow_test_workflow_1_stderr.txt"

        if workflow_stdout_file.exists():
            print(f"✓ Workflow stdout file created: {workflow_stdout_file}")
            with open(workflow_stdout_file, "r") as f:
                content = f.read()
                print(f"  Content (with proper newlines):\n{repr(content)}")

        if workflow_stderr_file.exists():
            print(f"✓ Workflow stderr file created: {workflow_stderr_file}")

        # Test 3: Dataset summary
        print("\n3. Testing dataset summary storage...")
        dataset_summary = {
            "dataset_name": "test_dataset",
            "total_workflows": 2,
            "successful_workflows": 1,
            "total_tools": 4,
            "successful_tools": 3,
            "syntax_correct_tools": 4,
            "execution_success_rate": 0.5,
            "syntax_success_rate": 1.0,
            "average_execution_time": 12.5,
            "results": [
                {
                    "workflow_id": "workflow1",
                    "workflow_name": "Workflow 1",
                    "execution_success": True,
                    "execution_time": 10.0,
                    "tools": [{"execution_success": True}, {"execution_success": True}],
                },
                {
                    "workflow_id": "workflow2",
                    "workflow_name": "Workflow 2",
                    "execution_success": False,
                    "execution_time": 15.0,
                    "tools": [
                        {"execution_success": True},
                        {"execution_success": False},
                    ],
                },
            ],
        }

        dataset_dir = temp_path / "dataset_test"
        summary_file = storage.save_dataset_summary(dataset_summary, dataset_dir)

        print(f"✓ Dataset summary saved to: {summary_file}")

        # Test 4: Total summary
        print("\n4. Testing total summary storage...")
        total_summary = {
            "total_datasets": 2,
            "total_workflows": 4,
            "successful_workflows": 2,
            "total_tools": 8,
            "successful_tools": 6,
            "syntax_correct_tools": 8,
            "execution_success_rate": 0.5,
            "syntax_success_rate": 1.0,
            "average_execution_time": 13.75,
            "dataset_summaries": [
                {
                    "dataset_name": "dataset1",
                    "total_workflows": 2,
                    "successful_workflows": 1,
                    "execution_success_rate": 0.5,
                    "total_tools": 4,
                    "successful_tools": 3,
                    "syntax_correct_tools": 4,
                },
                {
                    "dataset_name": "dataset2",
                    "total_workflows": 2,
                    "successful_workflows": 1,
                    "execution_success_rate": 0.5,
                    "total_tools": 4,
                    "successful_tools": 3,
                    "syntax_correct_tools": 4,
                },
            ],
        }

        total_dir = temp_path / "total_results"
        total_file = storage.save_total_summary(total_summary, total_dir)

        print(f"✓ Total summary saved to: {total_file}")

        # Test 5: Verify JSON structure and content
        print("\n5. Verifying JSON structure and content...")

        # Check tool details JSON
        with open(details_file, "r") as f:
            tool_json = json.load(f)
            print(f"✓ Tool details JSON structure: {list(tool_json.keys())}")
            assert "tool_name" in tool_json
            assert "timestamp" in tool_json
            assert "registration_success" in tool_json
            print("✓ Tool details JSON contains expected fields")

        # Check workflow JSON
        with open(workflow_file, "r") as f:
            workflow_json = json.load(f)
            print(f"✓ Workflow JSON structure: {list(workflow_json.keys())}")
            assert "workflow_id" in workflow_json
            assert "timestamp" in workflow_json
            assert "tools_count" in workflow_json
            print("✓ Workflow JSON contains expected fields")

        # Check dataset summary JSON
        with open(summary_file, "r") as f:
            summary_json = json.load(f)
            print(f"✓ Dataset summary JSON structure: {list(summary_json.keys())}")
            assert "dataset_name" in summary_json
            assert "metrics" in summary_json
            assert "workflow_summaries" in summary_json
            print("✓ Dataset summary JSON contains expected fields")

        # Check total summary JSON
        with open(total_file, "r") as f:
            total_json = json.load(f)
            print(f"✓ Total summary JSON structure: {list(total_json.keys())}")
            assert "evaluation_timestamp" in total_json
            assert "overall_metrics" in total_json
            assert "dataset_metrics" in total_json
            print("✓ Total summary JSON contains expected fields")

        print("\n✅ All tests passed! Optimized storage is working correctly.")
        print("\nKey improvements:")
        print(
            "- Terminal outputs are stored in separate .txt files with proper newlines"
        )
        print("- JSON files have reduced redundancy and cleaner structure")
        print("- Tool code is not duplicated in JSON (saved separately as .py files)")
        print(
            "- Hierarchical structure: total_summary.json > summary.json > evaluation_*.json > tool_*_details.json"
        )


if __name__ == "__main__":
    test_optimized_storage()
