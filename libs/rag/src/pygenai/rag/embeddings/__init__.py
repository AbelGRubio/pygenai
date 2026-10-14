"""Embedding providers for the RAG package.

This package centralizes how text is converted into vectors. Concrete adapters
can be implemented in dedicated modules while the public API remains stable and
shared across vector stores and retrievers.
"""

from __future__ import annotations

from .base import BaseEmbeddings
from .litellm import LiteLLMEmbeddings
from .simple import SimpleEmbeddings

__all__ = [
    "BaseEmbeddings",
    "LiteLLMEmbeddings",
    "SimpleEmbeddings",
]
