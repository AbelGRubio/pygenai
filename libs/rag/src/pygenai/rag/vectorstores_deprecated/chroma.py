"""ChromaDB vector store adapter.

========================================================================================================================
Name:         pygenai/rag/vectorstores/chroma.py
Description:  Offers a robust wrapper for Chroma collections with connection and similarity search.
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

    def __init__(
        self,
        collection_name: str = "pygenai",
        *,
        client: Any | None = None,
        persist_directory: str | None = None,
    ) -> None:
        """Initialize the Chroma client wrapper and target collection.

        Args:
            collection_name: Name of the Chroma collection.
            client: Optional pre-configured Chroma client instance.
            persist_directory: Optional filesystem path for persistent storage.
        """
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
        """Add documents and optional embeddings to a Chroma collection."""
        ids = [str(getattr(document, "id", f"chroma-{index}")) for index, document in enumerate(documents)]
        texts = [document.text for document in documents]
        metadatas = [dict(document.metadata) if document.metadata else {} for document in documents]

        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": texts,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = [list(emb) for emb in embeddings]

        self._collection.add(**kwargs)
        return ids

    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Query a Chroma collection for nearby chunks or documents."""
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": k,
        }
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
            # Chroma distances (e.g. L2) can be converted or used directly; smaller is closer.
            score = float(1.0 / (1.0 + distances[i])) if i < len(distances) else 0.0

            document = Document(id=doc_id, text=text, metadata=meta)
            results.append(SearchHit(document=document, score=score, metadata=meta))

        return results

    def delete(self, ids: Sequence[str]) -> None:
        """Delete items from Chroma by identifier."""
        self._collection.delete(ids=list(ids))


__all__ = ["ChromaVectorStore"]
