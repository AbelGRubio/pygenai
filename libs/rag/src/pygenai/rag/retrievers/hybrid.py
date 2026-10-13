"""Hybrid retrieval combining vector similarity and keyword signals.

========================================================================================================================
Name:         pygenai/rag/retrievers/hybrid.py
Description:  Merges vector relevance with lexical overlap to improve context discovery.
Project:      Pygenai
Date:         2026-10-02 12:19:24
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby

from pygenai.core.base_retriever import BaseRetriever, RetrievedDocument
from pygenai.core.data import Chunk, Document
from pygenai.rag.vectorstores.base import VectorStore


class HybridRetriever(BaseRetriever):
    """Fuse vector similarity with a lightweight keyword scoring strategy."""

    def __init__(self, vector_store: VectorStore, *, vector_weight: float = 0.7, keyword_weight: float = 0.3) -> None:
        """Initialize the hybrid ranking weights."""
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    @staticmethod
    def _keyword_score(query: str, document: Document | Chunk) -> float:
        """Compute a token-overlap score as a cheap BM25-like approximation."""
        query_tokens = {token.lower() for token in query.split() if token}
        if not query_tokens:
            return 0.0
        document_tokens = {token.lower() for token in document.text.split() if token}
        if not document_tokens:
            return 0.0
        overlap = len(query_tokens & document_tokens)
        return overlap / max(len(query_tokens), 1)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Return the top results after blending vector and keyword relevance."""
        vector_hits = self.vector_store.similarity_search(query, k=top_k)
        candidates: dict[str, RetrievedDocument] = {}

        for hit in vector_hits:
            document = hit.document
            key = getattr(document, "id", str(id(document)))
            keyword_score = self._keyword_score(query, document)
            combined_score = (self.vector_weight * float(hit.score)) + (self.keyword_weight * keyword_score)
            candidates[key] = RetrievedDocument(document=document, score=combined_score, metadata={**hit.metadata})

        for document in self._iter_documents_from_vector_hits(vector_hits):
            key = getattr(document, "id", str(id(document)))
            if key in candidates:
                continue
            keyword_score = self._keyword_score(query, document)
            candidates[key] = RetrievedDocument(
                document=document, score=self.keyword_weight * keyword_score, metadata={}
            )

        ranked = sorted(candidates.values(), key=lambda result: result.score, reverse=True)[:top_k]
        for index, result in enumerate(ranked, start=1):
            result.rank = index
        return ranked

    def retrieve_and_format(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """Retrieve documents and format them for downstream consumption."""
        retrieved_items = self.retrieve(query, top_k=top_k)
        docs_info = [
            (
                getattr(doc.document, "source", None)
                or (doc.metadata.get("source") if hasattr(doc, "metadata") else "unknown"),
                doc.document.text,
            )
            for item in retrieved_items
            for doc in (item if isinstance(item, list) else [item])
            if hasattr(doc, "document") and hasattr(doc.document, "text")
        ]

        # 2. Ordenamos por fuente (necesario para que groupby funcione correctamente)
        docs_info.sort(key=lambda x: x[0])

        # 3. Agrupamos por fuente y unimos sus textos usando una comprensión de lista final
        formatted_sources = [
            f"Source: {source}\nContent:\n" + "\n".join(text for _, text in group)
            for source, group in groupby(docs_info, key=lambda x: x[0])
        ]

        return formatted_sources

    @staticmethod
    def _iter_documents_from_vector_hits(hits: Sequence[object]) -> list[Document | Chunk]:
        """Extract document objects from SearchHit-like objects."""
        return [hit.document for hit in hits if hasattr(hit, "document")]


__all__ = ["HybridRetriever"]
