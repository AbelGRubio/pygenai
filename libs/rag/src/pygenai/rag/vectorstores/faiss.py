"""FAISS-backed vector store for local high-performance search.

========================================================================================================================
Name:         pygenai/rag/vectorstores/faiss.py
Description:  Wraps the FAISS vector database for local development and rapid vector retrieval with similarity search.
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

        import numpy as np

        float_matrix = np.array([list(values) for values in embeddings], dtype=np.float32)
        self._index.add(float_matrix)

        for document in documents:
            self._documents.append(document)
            self._ids.append(str(getattr(document, "id", f"vector-{len(self._ids)}")))

        return self._ids[-len(documents) :]

    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Find documents by querying the FAISS index.

        Note: FAISS operates purely on numerical vectors. This method requires a precomputed
        query embedding vector passed via filter_search or expects an external embedding step.
        """
        del filter_search
        query_embedding = filter_search.get("query_embedding") if filter_search else None
        if query_embedding is None:
            msg = (
                "FAISS similarity_search requires a query embedding vector via filter_search={'query_embedding': [...]}"
            )
            raise ValueError(msg)

        import numpy as np

        q_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self._index.search(q_vector, min(k, len(self._documents)))

        results: list[SearchHit] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            document = self._documents[idx]
            results.append(
                SearchHit(
                    document=document,
                    score=float(score),
                    metadata=dict(document.metadata) if document.metadata else {},
                )
            )
        return results

    def delete(self, ids: Sequence[str]) -> None:
        """FAISS standard flat indexes do not support granular item deletion directly."""
        del ids
        msg = "FAISS deletion is not supported natively in flat index wrappers."
        raise NotImplementedError(msg)


__all__ = ["FAISSVectorStore"]
