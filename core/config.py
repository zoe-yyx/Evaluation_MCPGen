"""
Evaluation Configuration Management
Configuration management for MCPFlow tools generation evaluation framework
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM interface"""
    provider: str = "openai"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_version: str = "2025-04-01-preview" 
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 2000
    temperature: float = 0.2
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def get(self, key: str, default=None):
        """Dictionary-style access support"""
        return getattr(self, key, default)


@dataclass
class E2EConfig:
    """Configuration for end-to-end execution evaluation"""

    execution_timeout: float
    max_tool_generation: int
    max_tool_generation_attempts: int
    runner: str

    def get(self, key: str, default=None):
        """Dictionary-style access support"""
        return getattr(self, key, default)


class EvaluationConfig:
    """Main evaluation configuration - requires YAML configuration file"""

    def __init__(self, config_path: str):
        """Initialize configuration from YAML file

        Args:
            config_path: Path to YAML configuration file (required)

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid or missing required fields
        """
        if not config_path:
            raise ValueError("Configuration file path is required")

        self.load_from_file(config_path)

    def get(self, key: str, default=None):
        """Dictionary-style access support"""
        return getattr(self, key, default)

    def load_from_file(self, config_path: str):
        """Load configuration from YAML file only"""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            # Allow .yaml, .yml, and .example files (which are YAML format)
            valid_extensions = [".yaml", ".yml"]
            is_example_file = str(config_file).endswith(".example")

            if (
                config_file.suffix.lower() not in valid_extensions
                and not is_example_file
            ):
                raise ValueError(
                    f"Only YAML configuration files are supported. Got: {config_file.suffix}"
                )

            import yaml

            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                raise ValueError("Configuration file is empty")

            self._load_from_dict(config_data)
            logger.info(f"Configuration loaded from {config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise

    def _load_from_dict(self, config_data: Dict[str, Any]):
        """Load configuration from dictionary with required field validation"""
        # Core settings (required)
        self.framework_version = self._get_required(config_data, "framework_version")
        self.evaluation_id = config_data.get("evaluation_id")

        # LLM configuration (required)
        llm_config = self._get_required(config_data, "llm_config")
        if not isinstance(llm_config, dict):
            raise ValueError("llm_config must be a dictionary")

        self.llm_config = LLMConfig(
            provider=self._get_required(llm_config, "provider"),
            model_name=self._get_required(llm_config, "model_name"),
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url"),
            temperature=self._get_required(llm_config, "temperature"),
            max_tokens=self._get_required(llm_config, "max_tokens"),
            timeout=self._get_required(llm_config, "timeout"),
            retry_attempts=self._get_required(llm_config, "retry_attempts"),
            retry_delay=self._get_required(llm_config, "retry_delay"),
        )

        # E2E configuration (required)
        e2e_config = self._get_required(config_data, "e2e_config")
        if not isinstance(e2e_config, dict):
            raise ValueError("e2e_config must be a dictionary")

        self.e2e_config = E2EConfig(
            execution_timeout=self._get_required(e2e_config, "execution_timeout"),
            max_tool_generation=self._get_required(e2e_config, "max_tool_generation"),
            max_tool_generation_attempts=self._get_required(
                e2e_config, "max_tool_generation_attempts"
            ),
            runner=self._get_required(e2e_config, "runner"),
        )

        # Other configurations (required)
        self.dataset_path = self._get_required(config_data, "dataset_path")
        self.output_path = self._get_required(config_data, "output_path")
        self.log_level = self._get_required(config_data, "log_level")
        self.debug_mode = self._get_required(config_data, "debug_mode")
        self.save_intermediate_results = self._get_required(
            config_data, "save_intermediate_results"
        )

    def _get_required(self, config_dict: Dict[str, Any], key: str) -> Any:
        """Get required configuration value with validation"""
        if key not in config_dict:
            raise ValueError(f"Required configuration key '{key}' is missing")

        value = config_dict[key]
        if value is None:
            raise ValueError(f"Required configuration key '{key}' cannot be None")

        return value

    def save_to_file(self, config_path: str):
        """Save current configuration to YAML file"""
        config_data = self.to_dict()
        config_file = Path(config_path)

        try:
            if config_file.suffix.lower() not in [".yaml", ".yml"]:
                raise ValueError(
                    f"Only YAML configuration files are supported. Got: {config_file.suffix}"
                )

            import yaml

            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {str(e)}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "framework_version": self.framework_version,
            "evaluation_id": self.evaluation_id,
            "llm_config": {
                "provider": self.llm_config.provider,
                "model_name": self.llm_config.model_name,
                "api_key": self.llm_config.api_key,
                "base_url": self.llm_config.base_url,
                "temperature": self.llm_config.temperature,
                "max_tokens": self.llm_config.max_tokens,
                "timeout": self.llm_config.timeout,
                "retry_attempts": self.llm_config.retry_attempts,
                "retry_delay": self.llm_config.retry_delay,
            },
            "e2e_config": {
                "execution_timeout": self.e2e_config.execution_timeout,
                "max_tool_generation": self.e2e_config.max_tool_generation,
                "max_tool_generation_attempts": self.e2e_config.max_tool_generation_attempts,
                "runner": self.e2e_config.runner,
            },
            "dataset_path": self.dataset_path,
            "output_path": self.output_path,
            "log_level": self.log_level,
            "debug_mode": self.debug_mode,
            "save_intermediate_results": self.save_intermediate_results,
        }

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a summary of evaluation settings"""
        return {
            "framework_version": self.framework_version,
            "llm_model": self.llm_config.model_name,
            "evaluation_tasks": {
                "tools_generation": {
                    "timeout": self.e2e_config.execution_timeout,
                    "max_tool_generation": self.e2e_config.max_tool_generation,
                },
            },
            "debug_mode": self.debug_mode,
        }
