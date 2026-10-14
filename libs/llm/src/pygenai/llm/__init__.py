"""Py LLM Module - Unified LLM Provider Abstraction.

========================================================================================================================
Name:         /llm/__init__.py
Description:  Main package exports for the LLM module
Project:      Py
Date:         2026-10-02 18:59:23
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from pygenai.llm.exceptions import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    LLMError,
    ModelNotFoundError,
    NoAvailableProvidersError,
    ProviderError,
    RateLimitError,
)
from pygenai.llm.litellm_provider import LiteLLMProvider

__all__ = [
    "AuthenticationError",
    "ConfigurationError",
    "InvalidResponseError",
    "LLMError",
    "LiteLLMProvider",
    "ModelNotFoundError",
    "NoAvailableProvidersError",
    "ProviderError",
    "RateLimitError",
]

__version__ = "0.1.0"
