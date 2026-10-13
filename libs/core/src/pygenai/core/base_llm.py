"""Base models and interfaces for LLM abstraction layer.

========================================================================================================================
Name:         genai/llm/base.py
Description:  Base classes, dataclasses, and interfaces for unified LLM provider abstraction
Project:      Pygenai
Date:         2026-10-02 18:59:23
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .models import Message


class LLMResponse(BaseModel):
    """Unified response model from any LLM provider."""

    content: str = Field(..., description="The generated text response from the model")
    model: str = Field(..., description="The model that generated the response")
    provider: str = Field(..., description="The provider that handled the request")
    usage: MessageUsage = Field(default_factory=lambda: MessageUsage(prompt_tokens=0, completion_tokens=0))
    cost: float = Field(default=0.0, description="Calculated cost of the API call in USD")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata from the provider")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class MessageUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(default=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Number of tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")

    def __init__(self, **data: Any) -> None:
        """Initialize MessageUsage and calculate total tokens if not provided."""
        super().__init__(**data)
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retry_on: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


@dataclass
class FallbackConfig:
    """Configuration for fallback strategy."""

    enabled: bool = True
    fallback_models: list[str] = field(default_factory=list)
    max_fallback_attempts: int = 3


@dataclass
class ProviderConfig:
    """Base configuration for any LLM provider."""

    name: str
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout: int = 30
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    fallback_config: FallbackConfig = field(default_factory=FallbackConfig)
    extra_params: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize the provider.

        Args:
            config: Provider configuration.
        """
        self.config = config
        self.name = config.name

    @abstractmethod
    async def generate(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of messages to send to the model.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse: The unified response from the model.

        Raises:
            ProviderError: If the provider fails to generate a response.
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the provider configuration.

        Returns:
            bool: True if configuration is valid, False otherwise.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Get list of available models for this provider.

        Returns:
            list[str]: List of model names.
        """
        pass
