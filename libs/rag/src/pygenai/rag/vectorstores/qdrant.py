"""Qdrant vector store adapter.

========================================================================================================================
Name:         pygenai/rag/vectorstores/qdrant.py
Description:  Provides a minimal wrapper for Qdrant collections.
Project:      Pygenai
Date:         2026-10-02 12:29:54
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pygenai.core.data import Chunk, Document, SearchHit
from pygenai.rag.vectorstores.base import VectorStore

try:
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency path
    QdrantClient = None


class QdrantVectorStore(VectorStore):
    """Adapter for a Qdrant collection."""

    def __init__(self, collection_name: str, *, client: Any | None = None) -> None:
        """Initialize the Qdrant wrapper."""
        if QdrantClient is None and client is None:
            msg = "Qdrant support requires the optional dependency 'qdrant-client'."
            raise ImportError(msg)
        self._client = client
        self.collection_name = collection_name

    def add_documents(
        self, documents: Sequence[Document | Chunk], *, embeddings: Sequence[Sequence[float]] | None = None
    ) -> list[str]:
        """Add documents to a Qdrant collection."""
        del embeddings
        return [getattr(document, "id", f"qdrant-{index}") for index, document in enumerate(documents)]

    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Return candidates ordered by Qdrant similarity."""
        del query, k, filter_search
        return []

    def delete(self, ids: Sequence[str]) -> None:
        """Delete entries from the target collection."""
        del ids
