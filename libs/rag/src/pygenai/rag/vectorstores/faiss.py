"""FAISS-backed vector store for local high-performance search.

========================================================================================================================
Name:         pygenai/rag/vectorstores/faiss.py
Description:  Wraps the FAISS vector database for local development and rapid vector retrieval.
Project:      Pygenai
Date:         2026-10-02 12:19:24
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
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency path
    faiss = None


class FAISSVectorStore(VectorStore):
    """Vector store integration for the FAISS library."""

    def __init__(self, dimension: int = 768, *, metric: str = "l2") -> None:
        """Initialize the FAISS index and supporting metadata."""
        if faiss is None:
            msg = "FAISS support requires the optional dependency 'faiss'."
            raise ImportError(msg)

        self.dimension = dimension
        self.metric = metric
        self._ids: list[str] = []
        self._documents: list[Document | Chunk] = []
        if metric == "l2":
            self._index = faiss.IndexFlatL2(dimension)
        else:
            self._index = faiss.IndexFlatIP(dimension)

    def add_documents(
        self, documents: Sequence[Document | Chunk], *, embeddings: Sequence[Sequence[float]] | None = None
    ) -> list[str]:
        """Add documents and embeddings to the FAISS index."""
        if embeddings is None:
            msg = "FAISS requires explicit embeddings to build the index."
            raise ValueError(msg)
        if len(documents) != len(embeddings):
            msg = "documents and embeddings must have the same length."
            raise ValueError(msg)

        float_matrix = [list(values) for values in embeddings]
        self._index.add(float_matrix)
        for document in documents:
            self._documents.append(document)
            self._ids.append(getattr(document, "id", f"vector-{len(self._ids)}"))
        return self._ids[-len(documents) :]

    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Find documents by querying the FAISS index using an embedding vector."""
        raise NotImplementedError("FAISS search requires the application layer to supply an embedding model.")

    def delete(self, ids: Sequence[str]) -> None:
        """Delete options are not implemented directly for FAISS; use a custom metadata index if needed."""
        del ids
        msg = "FAISS deletion is not implemented directly in this convenience wrapper."
        raise NotImplementedError(msg)


__all__ = ["FAISSVectorStore"]
