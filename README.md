
# MCPFlow-Evaluation

A comprehensive evaluation framework for testing LLM capabilities in regenerating MCP (Model Context Protocol) tools and validating their functionality within existing workflows.

## Overview

MCPFlow-Evaluation evaluates how well Large Language Models can regenerate functional MCP tools when given workflow context. The framework uses a robust approach focusing on multiple evaluation metrics: **syntax correctness**, **tool registration success**, **individual tool execution**, and **workflow integration success**.

## Key Features

- **LLM-Driven Tool Regeneration**: Regenerate MCP tools using LLM with rich workflow context
- **Comprehensive Evaluation Metrics**: Multi-level validation from syntax to workflow execution
- **Individual Tool Testing**: Each tool is tested independently for accurate assessment
- **FastMCP Integration**: Built on the FastMCP framework for MCP server/client implementation
- **Context-Aware Generation**: Uses workflow.json, server.py, run_workflow.py, and pyproject.toml for context
- **Multi-Dataset Support**: Process multiple datasets with independent environments
- **Hierarchical Results**: Organized output structure with dataset-level and overall summaries
- **Detailed Logging**: Complete test output capture for debugging and analysis

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd MCPFlow-Evaluation

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Configuration

Create your configuration file based on the provided example:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your settings. Here are the key configuration sections:

```yaml
# Framework Configuration
framework_version: "1.0.0"
evaluation_id: "default-evaluation"
output_format: "json"

# Dataset Configuration
dataset_path: "dataset/example"  # Path to dataset directory
tasks: ['tools_generation']      # Evaluation tasks to run

# Logging Configuration
log_level: "INFO"
debug_mode: false
save_intermediate_results: false

# LLM Configuration (Required)
llm_config:
  provider: "openai"
  model_name: "gpt-4o"
  api_key: "YOUR_API_KEY_HERE"  # Replace with your actual API key
  base_url: "http://localhost:4000/v1"  # Or OpenAI API URL
  temperature: 0.2
  max_tokens: 2000
  timeout: 60.0
  retry_attempts: 2
  retry_delay: 1.0

# End-to-end Execution Configuration
e2e_config:
  execution_timeout: 30.0
  max_tool_generation: 5
  max_tool_generation_attempts: 2
  generated_module_name: "generated_tools"
  allow_parallel_execution: false
  error_handling_strategy: "continue"
  validation_level: "import"
  runner: "uv"  # Use uv for workflow execution
```

### 3. Running Evaluation

Test the evaluation system:

```bash
# Run the MCP tool evaluation test
uv run python test/test_mcp_tool_evaluation.py
```

Test LLM connection:

```bash
# Verify LLM connectivity
uv run python test/test_llm_connection.py
```

## Architecture

### Core Components

- **[`tasks/mcp_tool_regenerator.py`](tasks/mcp_tool_regenerator.py)**: Core tool regeneration logic using LLM
- **[`tasks/mcp_tool_evaluator.py`](tasks/mcp_tool_evaluator.py)**: Dataset-level evaluation coordinator
- **[`tasks/result_storage.py`](tasks/result_storage.py)**: Result persistence and management
- **[`core/config.py`](core/config.py)**: Configuration management system
- **[`utils/llm_interface.py`](utils/llm_interface.py)**: LLM communication interface
- **[`utils/metrics.py`](utils/metrics.py)**: Metrics calculation utilities
- **[`models/workflow.py`](models/workflow.py)**: Workflow data models

### Evaluation Process

1. **Context Collection**: Gather workflow context from multiple sources:
   - `workflow.json`: Workflow structure and step descriptions
   - `server.py`: Server implementation and tool registration patterns
   - `run_workflow.py`: Tool usage examples and execution context
   - `pyproject.toml`: Available dependencies and project configuration

2. **Tool Regeneration**: Use LLM to regenerate each MCP tool with:
   - Rich contextual prompts
   - Explicit parameter requirements (no *args/**kwargs)
   - JSON-serializable return values
   - FastMCP compatibility

3. **Multi-Level Validation**:
   - **Syntax Validation**: Check generated code syntax using Python's `compile()` function
   - **Tool Registration**: Register regenerated tools with the MCP server using FastMCP SDK
   - **Individual Tool Testing**: Test each tool independently in isolated server sessions
   - **Workflow Integration**: Run original workflow tests to validate end-to-end functionality

4. **Results Management**: Generate comprehensive reports with:
   - Individual tool test results and outputs
   - Dataset-level summaries
   - Overall evaluation statistics
   - Detailed error logs and debugging information

## Dataset Structure

Datasets should follow this structure (see [`dataset/README.md`](dataset/README.md) for detailed specification):

```
dataset/
├── example/
│   ├── workflow.json          # Workflow definition
│   ├── run_workflow.py        # Workflow execution script
│   ├── pyproject.toml         # Dependencies and project config
│   ├── uv.lock               # Dependency lock file
│   └── mcp_server/
│       ├── server.py          # FastMCP server implementation
│       └── tools/             # Tool implementations
│           ├── __init__.py
│           ├── add.py
│           ├── multiply.py
│           └── ...
```

### Required Files

- **`workflow.json`**: Workflow structure with step descriptions and tool mappings
- **`run_workflow.py`**: Executable workflow using FastMCP Client
- **`mcp_server/server.py`**: FastMCP server that registers tools
- **`mcp_server/tools/*.py`**: Individual tool implementations
- **`pyproject.toml`**: Project dependencies and configuration
- **`uv.lock`**: Dependency lock file for reproducible environments

### Tool Requirements

Tools must be compatible with FastMCP:

```python
def tool_name(param1: str, param2: int) -> dict:
    """Tool description"""
    # Implementation
    return {"result": "value"}

# TOOL_METADATA: {"name": "tool_name", "description": "...", "parameters": {...}}
```

**Important**: Tools cannot use `*args` or `**kwargs` (FastMCP limitation).

## Results Structure

The framework generates organized results in the following structure:

```
results/
├── dataset_name/
│   ├── evaluation_workflow_name.json    # Detailed workflow results
│   ├── summary.json                     # Dataset-level summary
│   ├── tool1.py                        # Regenerated tool files
│   ├── tool2.py
│   └── ...
└── overall_summary.json                 # Cross-dataset summary
```

### Result Files

- **`evaluation_*.json`**: Complete evaluation results for each workflow including:
  - Individual tool test results and outputs
  - Syntax validation results
  - Registration success status
  - Execution test logs
  - Error messages and debugging information

- **`summary.json`**: Dataset-level statistics including:
  - Success rates for each validation level
  - Execution time metrics
  - Tool-by-tool breakdown

- **`overall_summary.json`**:
- **`overall_summary.json`**: Cross-dataset aggregated statistics

## Evaluation Metrics

The framework provides comprehensive evaluation across multiple levels:

### 1. Syntax Correctness Rate
- Percentage of generated tools with valid Python syntax
- Validated using Python's built-in `compile()` function
- Indicates LLM's ability to generate syntactically correct code

### 2. Tool Registration Success Rate
- Percentage of tools that successfully register with the MCP server
- Tests FastMCP compatibility and metadata correctness
- Indicates proper tool interface implementation

### 3. Individual Tool Execution Success Rate
- Percentage of tools that execute successfully in isolation
- Each tool is tested independently in its own server session
- Provides accurate assessment of individual tool functionality

### 4. Workflow Integration Success Rate
- Percentage of workflows that execute successfully with regenerated tools
- Tests end-to-end functionality and tool interaction
- Indicates practical utility of generated tools in real scenarios

## Example Results

```json
{
  "dataset_name": "example",
  "total_workflows": 1,
  "successful_workflows": 1,
  "total_tools": 5,
  "successful_tools": 5,
  "syntax_correct_tools": 5,
  "execution_success_rate": 1.0,
  "syntax_success_rate": 1.0,
  "average_execution_time": 12.60,
  "results": [
    {
      "workflow_id": "simple_math_operations_workflow",
      "workflow_name": "Simple Math Operations Workflow",
      "tools": [
        {
          "name": "add",
          "registration_success": true,
          "execution_success": true,
          "syntax_correct": true,
          "test_stderr": "... detailed test output ..."
        }
      ],
      "execution_success": true,
      "execution_time": 12.60
    }
  ]
}
```

Console output example:
```
=== Evaluation Results ===
Dataset: example
Total workflows: 1
Successful workflows: 1
Total tools: 5
Tools with correct syntax: 5
Tools that passed registration: 5
Tools that passed individual tests: 5
Workflow success rate: 100.00%
Syntax correctness rate: 100.00%
Registration success rate: 100.00%
Individual test success rate: 100.00%
Average execution time: 12.60s

--- Workflow: Simple Math Operations Workflow ---
  Overall workflow success: True
  Tools regenerated: 5
    - add: ✓ Syntax OK | ✓ Registration OK | ✓ Individual Test OK
    - subtract: ✓ Syntax OK | ✓ Registration OK | ✓ Individual Test OK
    - multiply: ✓ Syntax OK | ✓ Registration OK | ✓ Individual Test OK
    - divide: ✓ Syntax OK | ✓ Registration OK | ✓ Individual Test OK
    - fetch_data: ✓ Syntax OK | ✓ Registration OK | ✓ Individual Test OK
```

## Development

### Project Structure

```
MCPFlow-Evaluation/
├── core/                      # Core framework components
│   └── config.py             # Configuration management
├── tasks/                     # Main evaluation tasks
│   ├── mcp_tool_regenerator.py  # Tool regeneration logic
│   ├── mcp_tool_evaluator.py    # Dataset evaluation coordinator
│   └── result_storage.py        # Result persistence
├── utils/                     # Utility modules
│   ├── llm_interface.py      # LLM communication
│   ├── logger.py             # Logging utilities
│   ├── metrics.py            # Metrics calculation
│   └── json_encoder.py       # JSON serialization utilities
├── models/                    # Data models
│   └── workflow.py           # Workflow representations
├── test/                      # Test suite
│   ├── test_mcp_tool_evaluation.py  # Main evaluation test
│   └── test_llm_connection.py       # LLM connectivity test
├── dataset/                   # Evaluation datasets
│   └── example/              # Example workflow
├── results/                   # Evaluation results (generated)
│   └── example/              # Dataset-specific results
└── docs/                      # Documentation
    └── mcp_tool_evaluation.md # Detailed evaluation docs
```

### Running Tests

```bash
# Run main evaluation test
uv run python test/test_mcp_tool_evaluation.py

# Test LLM connection
uv run python test/test_llm_connection.py

# Run with specific configuration
uv run python test/test_mcp_tool_evaluation.py --config custom_config.yaml
```

### Adding New Datasets

1. Create a new directory under `dataset/`
2. Follow the structure specified in [`dataset/README.md`](dataset/README.md)
3. Ensure all required files are present and properly formatted
4. Include `pyproject.toml` and `uv.lock` for dependency management
5. Test locally before adding to evaluation suite

### Configuration Options

The framework supports extensive configuration through `config.yaml`:

#### Framework Settings
- `framework_version`: Version identifier
- `evaluation_id`: Unique identifier for evaluation runs
- `output_format`: Result output format (currently JSON only)

#### Dataset Settings
- `dataset_path`: Path to dataset directory or specific dataset
- `tasks`: List of evaluation tasks to execute

#### LLM Settings
- `provider`: LLM provider (openai, azure, etc.)
- `model_name`: Specific model to use
- `api_key`: Authentication key
- `base_url`: API endpoint URL
- `temperature`: Generation randomness (0.0-1.0)
- `max_tokens`: Maximum response length
- `timeout`: Request timeout in seconds
- `retry_attempts`: Number of retry attempts
- `retry_delay`: Delay between retries

#### Execution Settings
- `execution_timeout`: Workflow execution timeout
- `max_tool_generation`: Maximum tools to generate
- `max_tool_generation_attempts`: Retry attempts for tool generation
- `validation_level`: Validation depth (syntax, import, runtime, integration)
- `runner`: Execution environment (python or uv)
- `error_handling_strategy`: How to handle errors (continue, stop, retry)

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the existing patterns
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## Troubleshooting

### Common Issues

**Configuration Errors**:
```
ValueError: Required configuration key 'api_key' is missing
```
- Solution: Ensure all required fields are present in your `config.yaml`

**Import Errors**:
```
ModuleNotFoundError: No module named 'tasks'
```
- Solution: Use `uv run` to execute scripts with proper environment

**LLM Connection Issues**:
```
Error code: 400 - {'error': {'message': 'Invalid API key'}}
```
- Solution: Verify your API key and LLM server configuration

**Tool Generation Issues**:
```
SyntaxError: invalid syntax in generated code
```
- Solution: Check LLM model capabilities and prompt engineering

**UV Environment Issues**:
```
uv: command not found
```
- Solution: Install uv package manager or use python runner in config

### Debug Mode

Enable debug mode in your configuration for detailed logging:

```yaml
debug_mode: true
log_level: "DEBUG"
save_intermediate_results: true
```

This will provide:
- Detailed LLM request/response logs
- Step-by-step execution traces
- Intermediate file saves
- Enhanced error reporting

### Performance Optimization

- **Parallel Execution**: Enable `allow_parallel_execution` for faster processing
- **Timeout Tuning**: Adjust `execution_timeout` based on workflow complexity
- **Retry Strategy**: Configure `retry_attempts` and `retry_delay` for reliability
- **Validation Level**: Use appropriate `validation_level` for your needs

## Security Considerations

- **API Keys**: Store in configuration files, never in code
- **Configuration Files**: Use `.
gitignore` to exclude sensitive files
- **Code Execution**: Generated tools run in isolated environments
- **Input Validation**: Validate all user inputs and configuration parameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastMCP framework for MCP server implementation
- UV package manager for Python environment management
- OpenAI and other LLM providers for tool generation capabilities

---

For more detailed information about the evaluation methodology and implementation details, see [`docs/mcp_tool_evaluation.md`](docs/mcp_tool_evaluation.md).