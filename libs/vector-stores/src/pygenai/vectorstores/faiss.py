"""FAISS-backed vector store for local high-performance search and memory.

========================================================================================================================
Name:         pygenai/vectorstores/faiss.py
Description:  Wraps the FAISS vector database for local development, document retrieval, and agent memory management.
Project:      Pygenai
Date:         2026-10-22 17:30:00
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pygenai.core.data import Chunk, Document, SearchHit
from .base import VectorStore

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    faiss = None


class FAISSVectorStore(VectorStore):
    """Vector store integration for the FAISS library supporting local RAG and memory."""

    def __init__(self, dimension: int = 768, *, metric: str = "l2") -> None:
        """Initialize the FAISS index and supporting metadata storage."""
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

        return self._ids[-len(documents):]

    def similarity_search(
            self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Find documents or memories by querying the FAISS index.

        Note: FAISS operates purely on numerical vectors. Requires a precomputed
        query embedding vector passed via filter_search={'query_vector': [...]}.
        Optional metadata filters (memory_type, user_id, session_id) are applied post-search.
        """
        del query
        query_vector = filter_search.get("query_vector") if filter_search else None
        if query_vector is None:
            msg = "FAISS similarity_search requires a query vector via filter_search={'query_vector': [...]}"
            raise ValueError(msg)

        import numpy as np

        q_vector = np.array([query_vector], dtype=np.float32)
        # Search more items if filtering is needed
        fetch_k = min(max(k * 3, 20), len(self._documents)) if self._documents else k
        distances, indices = self._index.search(q_vector, min(fetch_k, len(self._documents)))

        results: list[SearchHit] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            document = self._documents[idx]
            metadata = dict(document.metadata) if document.metadata else {}

            # Apply manual post-filtering for memory attributes if specified
            if filter_search:
                match = True
                for key in ["memory_type", "user_id", "session_id"]:
                    if key in filter_search and metadata.get(key) != filter_search[key]:
                        match = False
                        break
                if not match:
                    continue

            results.append(
                SearchHit(
                    document=document,
                    score=float(score),
                    metadata=metadata,
                )
            )
            if len(results) >= k:
                break

        return results

    def delete(self, ids: Sequence[str]) -> None:
        """Remove entries by filtering out matching IDs from the local document registry."""
        id_set = set(ids)
        new_docs: list[Document | Chunk] = []
        new_ids: list[str] = []

        import numpy as np
        kept_embeddings = []

        for i, doc_id in enumerate(self._ids):
            if doc_id not in id_set:
                new_docs.append(self._documents[i])
                new_ids.append(doc_id)
                # Rebuild index or maintain vectors if required.
                # For flat index wrappers, rebuilding is a robust approach on deletion:
                # (Alternatively, store raw vectors alongside documents if frequent deletions are expected).

        self._documents = new_docs
        self._ids = new_ids

        # Re-initialize index
        if self.metric == "l2":
            self._index = faiss.IndexFlatL2(self.dimension)
        else:
            self._index = faiss.IndexFlatIP(self.dimension)


__all__ = ["FAISSVectorStore"]