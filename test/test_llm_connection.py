#!/usr/bin/env python
"""
Test script for LLM API connection

This script tests the connection to the LLM API using the configuration in config.yaml.
"""

import asyncio
import logging
import sys
from pathlib import Path

from core.config import EvaluationConfig
from utils.llm_interface import LLMInterface
from utils.logger import setup_logger

logger = setup_logger(__name__)


async def test_llm_connection():
    """Test the connection to the LLM API"""

    # Load configuration using the new secure configuration system
    try:
        config_path = "config.yaml"
        if not Path(config_path).exists():
            config_path = "config.yaml.example"
            if not Path(config_path).exists():
                logger.error(
                    "No configuration file found. Please create config.yaml based on config.yaml.example"
                )
                return

        # Use the new EvaluationConfig class
        config = EvaluationConfig(config_path)
        llm_config = config.llm_config

        logger.info(
            f"Testing connection to {llm_config.provider} API with model {llm_config.model_name}"
        )
        logger.info(f"Base URL: {llm_config.base_url}")

        # Check if API key is set
        if not llm_config.api_key or llm_config.api_key == "YOUR_API_KEY_HERE":
            logger.error(
                "API key is not set in config.yaml. Please set a valid API key."
            )
            return

        # Create LLM interface
        llm = LLMInterface(llm_config)

        # Test connection with a simple prompt
        logger.info("Sending test request to LLM API...")
        response = await llm.generate_response(
            "Generate a simple Python function that adds two numbers.",
            system_message="You are a helpful assistant that writes clean Python code.",
            max_tokens=100,
        )

        logger.info("LLM API connection successful!")
        logger.info(f"Response: {response}")

    except Exception as e:
        logger.error(f"Error testing LLM connection: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_llm_connection())
