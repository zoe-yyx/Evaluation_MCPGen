# utils/llm_interface.py

import asyncio
import logging
from typing import Optional, List
import openai
from openai import AsyncOpenAI, AsyncAzureOpenAI
from core.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMInterface:
    """Interface for interacting with various LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Setup LLM client based on provider"""
        if self.config.provider == "openai":
            self.client = AsyncOpenAI(
                api_key=self.config.api_key, 
                base_url=self.config.base_url
            )
        elif self.config.provider == "azure":
            # Azure OpenAI 配置
            # 使用方式:
            #   provider: azure
            #   api_key: your-azure-api-key
            #   base_url: https://gpt.yunstorm.com/
            #   api_version: 2025-04-01-preview
            #   model_name: gpt-4o (或 gpt-4o-mini, o1, o3-mini, o3, o4-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano)
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.config.base_url,
                api_key=self.config.api_key,
                api_version=getattr(self.config, 'api_version', '2025-04-01-preview'),
            )
            logger.info(f"Azure OpenAI client initialized: endpoint={self.config.base_url}, model={self.config.model_name}")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    async def generate_response(
        self, prompt: str, system_message: Optional[str] = None, **kwargs
    ) -> str:
        """
        Generate response from LLM

        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        for attempt in range(self.config.retry_attempts):
            try:
                messages = []

                if system_message:
                    messages.append({"role": "system", "content": system_message})

                messages.append({"role": "user", "content": prompt})

                # Prepare parameters
                params = {
                    "model": self.config.model_name,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                }

                logger.debug(f"LLM request: provider={self.config.provider}, model={params['model']}, "
                           f"temperature={params['temperature']}, max_tokens={params['max_tokens']}")

                # Both OpenAI and Azure use the same chat completions API
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**params),
                    timeout=self.config.timeout,
                )
                
                content = response.choices[0].message.content
                logger.debug(f"LLM response received: {len(content) if content else 0} chars")
                return content

            except asyncio.TimeoutError:
                logger.warning(f"LLM request timeout on attempt {attempt + 1}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

            except Exception as e:
                logger.warning(f"LLM request failed on attempt {attempt + 1}: {str(e)}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise

        raise RuntimeError("All LLM request attempts failed")

    async def generate_batch_responses(
        self, prompts: List[str], system_message: Optional[str] = None, **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts"""

        tasks = [
            self.generate_response(prompt, system_message, **kwargs)
            for prompt in prompts
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(
                    f"Failed to generate response for prompt {i}: {str(response)}"
                )
                results.append("")
            else:
                results.append(response)

        return results

    def validate_connection(self) -> bool:
        """Validate LLM connection and configuration"""
        try:
            # Simple test request
            test_response = asyncio.run(
                self.generate_response("Test connection", max_tokens=10)
            )
            return test_response is not None and len(test_response) > 0

        except Exception as e:
            logger.error(f"LLM connection validation failed: {str(e)}")
            return False