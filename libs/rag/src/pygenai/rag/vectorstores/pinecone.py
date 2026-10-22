"""Pinecone vector store adapter.

========================================================================================================================
Name:         pygenai/rag/vectorstores/pinecone.py
Description:  Provides a robust wrapper for Pinecone-backed vector storage with connection and similarity search.
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
    from pinecone import Pinecone  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency path
    Pinecone = None


class PineconeVectorStore(VectorStore):
    """Adapter for Pinecone vector storage."""

    def __init__(
        self,
        index_name: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        namespace: str = "",
    ) -> None:
        """Initialize the Pinecone integration and target index connection."""
        if Pinecone is None and client is None:
            msg = "Pinecone support requires the optional dependency 'pinecone'."
            raise ImportError(msg)

        if client is not None:
            self._client = client
        else:
            self._client = Pinecone(api_key=api_key)

        self.index_name = index_name
        self.namespace = namespace
        self._index = self._client.Index(index_name)

    def add_documents(
        self, documents: Sequence[Document | Chunk], *, embeddings: Sequence[Sequence[float]] | None = None
    ) -> list[str]:
        """Add documents with embeddings to the target Pinecone index."""
        if embeddings is None:
            msg = "Pinecone requires explicit embeddings to upsert vectors."
            raise ValueError(msg)
        if len(documents) != len(embeddings):
            msg = "documents and embeddings must have the same length."
            raise ValueError(msg)

        vectors = []
        ids = []
        for i, document in enumerate(documents):
            doc_id = str(getattr(document, "id", f"pinecone-{i}"))
            ids.append(doc_id)
            metadata = dict(document.metadata) if document.metadata else {}
            metadata["text"] = document.text
            vectors.append({
                "id": doc_id,
                "values": list(embeddings[i]),
                "metadata": metadata,
            })

        self._index.upsert(vectors=vectors, namespace=self.namespace)
        return ids

    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Query a Pinecone index for close vectors using a query vector."""
        del query
        if filter_search is None or "query_vector" not in filter_search:
            msg = "Pinecone similarity_search requires a query vector via filter_search={'query_vector': [...]}"
            raise ValueError(msg)

        query_vector = filter_search["query_vector"]
        pinecone_filter = filter_search.get("filter")

        response = self._index.query(
            vector=list(query_vector),
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
            filter=pinecone_filter,
        )

        results: list[SearchHit] = []
        matches = response.get("matches", [])
        for match in matches:
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")
            doc_id = match.get("id", "")
            score = float(match.get("score", 0.0))

            document = Document(id=doc_id, text=text, metadata=metadata)
            results.append(SearchHit(document=document, score=score, metadata=metadata))

        return results

    def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors from a Pinecone index."""
        self._index.delete(ids=list(ids), namespace=self.namespace)


__all__ = ["PineconeVectorStore"]