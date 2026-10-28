"""ChromaDB vector store adapter.

========================================================================================================================
Name:         pygenai/vectorstores/chroma.py
Description:  Offers a robust wrapper for Chroma collections with connection, file storage, and memory filtering.
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
    import chromadb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    chromadb = None


class ChromaVectorStore(VectorStore):
    """Adapter for ChromaDB vector storage supporting RAG documents and memory records."""

    def __init__(
        self,
        collection_name: str = "pygenai",
        *,
        client: Any | None = None,
        persist_directory: str | None = None,
    ) -> None:
        """Initialize the Chroma vector store with a collection name and optional client configuration."""
        if chromadb is None and client is None:
            msg = "Chroma support requires the optional dependency 'chromadb'."
            raise ImportError(msg)

        if client is not None:
            self._client = client
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self.collection_name = collection_name
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_documents(
        self, documents: Sequence[Document | Chunk], *, embeddings: Sequence[Sequence[float]] | None = None
    ) -> list[str]:
        """Add a collection of documents to the Chroma collection with optional precomputed embeddings."""
        ids = [str(getattr(doc, "id", f"chroma-{i}")) for i, doc in enumerate(documents)]
        texts = [doc.text for doc in documents]
        metadatas = [dict(doc.metadata) if doc.metadata else {} for doc in documents]

        kwargs: dict[str, Any] = {"ids": ids, "documents": texts, "metadatas": metadatas}
        if embeddings is not None:
            kwargs["embeddings"] = [list(emb) for emb in embeddings]

        self._collection.add(**kwargs)
        return ids

    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Perform a similarity search in the Chroma collection using a query string and optional filtering."""
        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": k}
        if filter_search is not None:
            kwargs["where"] = filter_search

        response = self._collection.query(**kwargs)
        results: list[SearchHit] = []

        if not response or not response.get("documents") or not response["documents"][0]:
            return results

        docs = response["documents"][0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        ids = response.get("ids", [[]])[0]

        for i, text in enumerate(docs):
            meta = metas[i] if i < len(metas) and metas[i] is not None else {}
            doc_id = ids[i] if i < len(ids) else f"chroma-hit-{i}"
            score = float(1.0 / (1.0 + distances[i])) if i < len(distances) else 0.0

            document = Document(id=doc_id, text=text, metadata=meta)
            results.append(SearchHit(document=document, score=score, metadata=meta))

        return results

    def delete(self, ids: Sequence[str]) -> None:
        """Delete documents from the Chroma collection by their identifiers."""
        self._collection.delete(ids=list(ids))


__all__ = ["ChromaVectorStore"]
