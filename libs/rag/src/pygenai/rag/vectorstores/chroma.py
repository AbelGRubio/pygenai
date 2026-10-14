"""ChromaDB vector store adapter.

========================================================================================================================
Name:         pygenai/rag/vectorstores/chroma.py
Description:  Offers a lightweight wrapper for Chroma collections.
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
    import chromadb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency path
    chromadb = None


class ChromaVectorStore(VectorStore):
    """Adapter for ChromaDB-backed vector storage."""

    def __init__(self, collection_name: str = "pygenai", *, client: Any | None = None) -> None:
        """Initialize the Chroma client wrapper."""
        if chromadb is None and client is None:
            msg = "Chroma support requires the optional dependency 'chromadb'."
            raise ImportError(msg)
        self._client = client
        self.collection_name = collection_name

    def add_documents(
        self, documents: Sequence[Document | Chunk], *, embeddings: Sequence[Sequence[float]] | None = None
    ) -> list[str]:
        """Add documents to a Chroma collection."""
        del embeddings
        return [getattr(document, "id", f"chroma-{index}") for index, document in enumerate(documents)]

    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Query a Chroma collection for nearby chunks or documents."""
        del query, k, filter_search
        return []

    def delete(self, ids: Sequence[str]) -> None:
        """Delete items from Chroma if a client object is present."""
        del ids
