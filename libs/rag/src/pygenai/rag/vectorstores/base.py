"""Base abstractions for vector databases.

========================================================================================================================
Name:         pygenai/rag/vectorstores/base.py
Description:  Defines a common API for storing embeddings and querying similarity.
Project:      Pygenai
Date:         2026-10-02 12:19:24
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pygenai.core.data import Chunk, Document, SearchHit


class VectorStore(ABC):
    """Abstract interface used by retrievers to query embeddings."""

    @abstractmethod
    def add_documents(
        self, documents: Sequence[Document | Chunk], *, embeddings: Sequence[Sequence[float]] | None = None
    ) -> list[str]:
        """Add a collection of documents and optional precomputed embeddings."""

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Return the best matching documents for a query."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Remove documents by identifier."""


__all__ = ["VectorStore"]
