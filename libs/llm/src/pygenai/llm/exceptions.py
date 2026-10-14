"""Custom exceptions for LLM module.

========================================================================================================================
Name:         genai/llm/exceptions.py
Description:  Custom exception classes for LLM operations, provider errors, and rate limiting
Project:      Pygenai
Date:         2026-10-02 18:59:23
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    pass


class ProviderError(LLMError):
    """Exception raised when a provider fails."""

    def __init__(self, provider: str, message: str, original_error: Exception | None = None) -> None:
        """Initialize the exception."""
        self.provider = provider
        self.message = message
        self.original_error = original_error
        super().__init__(f"Provider {provider} error: {message}")


class RateLimitError(LLMError):
    """Exception raised when rate limit is exceeded (429 error)."""

    def __init__(self, provider: str, retry_after: int | None = None) -> None:
        """Initialize the exception."""
        self.provider = provider
        self.retry_after = retry_after
        message = f"Rate limit exceeded for provider {provider}"
        if retry_after:
            message += f" (retry after {retry_after}s)"
        super().__init__(message)


class AuthenticationError(LLMError):
    """Exception raised when authentication fails."""

    def __init__(self, provider: str, message: str = "Invalid credentials") -> None:
        """Initialize the exception."""
        self.provider = provider
        super().__init__(f"Authentication failed for provider {provider}: {message}")


class ModelNotFoundError(LLMError):
    """Exception raised when a model is not available."""

    def __init__(self, model: str, provider: str | None = None) -> None:
        """Initialize the exception."""
        self.model = model
        self.provider = provider
        msg = f"Model '{model}' not found"
        if provider:
            msg += f" for provider '{provider}'"
        super().__init__(msg)


class NoAvailableProvidersError(LLMError):
    """Exception raised when no providers are available (all fallbacks exhausted)."""

    def __init__(self, attempted_providers: list[str]) -> None:
        """Initialize the exception."""
        self.attempted_providers = attempted_providers
        super().__init__(f"No available providers. Attempted: {', '.join(attempted_providers)}")


class ConfigurationError(LLMError):
    """Exception raised when configuration is invalid."""

    def __init__(self, message: str) -> None:
        """Initialize the exception."""
        super().__init__(f"Configuration error: {message}")


class InvalidResponseError(LLMError):
    """Exception raised when provider response is invalid or malformed."""

    def __init__(self, provider: str, message: str) -> None:
        """Initialize the exception."""
        self.provider = provider
        super().__init__(f"Invalid response from {provider}: {message}")
