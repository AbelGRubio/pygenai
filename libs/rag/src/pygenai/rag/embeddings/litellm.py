"""LiteLLM-backed embedding provider.

========================================================================================================================
Name:         pygenai/rag/embeddings/litellm.py
Description:  Adapter for LiteLLM embeddings using LangChain's LiteLLMEmbeddings client.
Project:      Pygenai
Date:         2026-10-06
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import BaseEmbeddings


class LiteLLMEmbeddings(BaseEmbeddings):
    """Thin wrapper over LangChain's LiteLLM embeddings implementation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini/gemini-embedding-2",
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the underlying LiteLLM embeddings client.

        Args:
            api_key: API key for the provider. When omitted, LiteLLM reads the
                standard environment variables.
            model: Name of the embedding model to call (e.g., openai/gemini/gemini-embedding-2).
            base_url: Optional custom API endpoint base URL.
            **kwargs: Extra keyword arguments forwarded to LangChain's LiteLLM embeddings client.
        """
        params: dict[str, Any] = {"model": model, **kwargs}
        if api_key is not None:
            params["api_key"] = api_key
        if base_url is not None:
            params["api_base"] = base_url
        super().__init__(**params)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embeddings for a sequence of document texts."""
        return super().embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding for a single query string."""
        return super().embed_query(text)
