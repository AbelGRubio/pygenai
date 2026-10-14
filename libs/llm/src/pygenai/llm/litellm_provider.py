"""LiteLLM unified provider implementation.

========================================================================================================================
Name:         genai/llm/providers/litellm_provider.py
Description:  LLM provider using LiteLLM Proxy / Gateway architecture
Project:      Pygenai
Date:         2026-10-04
Status:       Refactored
========================================================================================================================
"""

from __future__ import annotations

import os
from typing import Any

import litellm

from pygenai.core.base_llm import LLMProvider, LLMResponse, Message, MessageUsage, ProviderConfig
from pygenai.llm.exceptions import AuthenticationError, ConfigurationError, InvalidResponseError, ProviderError


class LiteLLMProvider(LLMProvider):
    """Unified LLM provider implementation using LiteLLM."""

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize LiteLLM provider.

        Args:
            config: Provider configuration.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        super().__init__(config)
        self.validate_config()

        self.api_key = config.api_key or os.getenv("LITELLM_API_KEY")
        if not self.api_key:
            raise AuthenticationError("litellm", "API key not provided in config or environment")

        if hasattr(config, "api_base") and config.api_base:
            litellm.api_base = config.api_base

    def validate_config(self) -> bool:
        """Validate configuration.

        Returns:
            bool: True if valid.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if not self.config.model:
            raise ConfigurationError("LiteLLM provider requires 'model' to be specified")
        return True

    async def generate(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Generate response using LiteLLM asynchronously.

        Args:
            messages: List of messages.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse: Unified response.

        Raises:
            ProviderError: If generation fails.
        """
        try:
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            response = await litellm.acompletion(
                model=self.config.model,
                messages=formatted_messages,
                api_key=self.api_key,
                timeout=self.config.timeout,
                num_retries=getattr(self.config, "retry_config", {}).get("max_retries", 2),
                **kwargs,
            )

            if not response or not hasattr(response, "choices"):
                raise InvalidResponseError("litellm", "Empty or malformed response")

            content = response.choices[0].message.content
            usage = MessageUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

            cost = litellm.completion_cost(
                model=self.config.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider="litellm",
                usage=usage,
                cost=cost,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "raw_response": str(response),
                },
            )

        except Exception as e:
            raise ProviderError("litellm", str(e), e) from e
