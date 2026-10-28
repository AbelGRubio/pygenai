"""Qdrant vector store adapter.

========================================================================================================================
Name:         pygenai/vectorstores/qdrant.py
Description:  Provides a robust wrapper for Qdrant collections with filtering for document retrieval and memory stores.
Project:      Pygenai
Date:         2026-10-22 17:16:00
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
    from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    from qdrant_client.http import models  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    QdrantClient = None
    models = None


class QdrantVectorStore(VectorStore):
    """Adapter for Qdrant collection supporting flexible payload filtering."""

    def __init__(
        self,
        collection_name: str,
        *,
        url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        prefer_grpc: bool = False,
    ) -> None:
        if QdrantClient is None and client is None:
            msg = "Qdrant support requires the optional dependency 'qdrant-client'."
            raise ImportError(msg)

        if client is not None:
            self._client = client
        else:
            self._client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)

        self.collection_name = collection_name

    def add_documents(
        self, documents: Sequence[Document | Chunk], *, embeddings: Sequence[Sequence[float]] | None = None
    ) -> list[str]:
        if embeddings is None:
            msg = "Qdrant requires explicit embeddings to upsert points."
            raise ValueError(msg)
        if len(documents) != len(embeddings):
            msg = "documents and embeddings must have the same length."
            raise ValueError(msg)

        points = []
        ids = []
        for i, document in enumerate(documents):
            point_id = i + 1
            str_id = str(getattr(document, "id", f"qdrant-{i}"))
            ids.append(str_id)

            payload = dict(document.metadata) if document.metadata else {}
            payload["text"] = document.text
            payload["original_id"] = str_id

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=list(embeddings[i]),
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self.collection_name, points=points)
        return ids

    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        del query
        if filter_search is None or "query_vector" not in filter_search:
            msg = "Qdrant similarity_search requires a query vector via filter_search={'query_vector': [...]}"
            raise ValueError(msg)

        query_vector = filter_search["query_vector"]
        qdrant_filter = filter_search.get("filter")

        hits = self._client.search(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            limit=k,
            query_filter=qdrant_filter,
        )

        results: list[SearchHit] = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.pop("text", "")
            original_id = payload.pop("original_id", str(hit.id))
            score = float(hit.score)

            document = Document(id=original_id, text=text, metadata=payload)
            results.append(SearchHit(document=document, score=score, metadata=payload))

        return results

    def delete(self, ids: Sequence[str]) -> None:
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="original_id", match=models.MatchAny(any=list(ids)))]
                )
            ),
        )


__all__ = ["QdrantVectorStore"]
