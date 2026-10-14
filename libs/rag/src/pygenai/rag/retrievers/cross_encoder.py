"""Cross-encoder-like reranker.

========================================================================================================================
Name:         pygenai/rag/retrievers/cross_encoder.py
Description:  Reorders retrieval candidates via lexical overlap with the query.
Project:      Pygenai
Date:         2026-10-02 12:29:54
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from pygenai.core.base_retriever import RetrievedDocument
from pygenai.core.data import Chunk, Document
from pygenai.rag.retrievers.rerank import Reranker


class CrossEncoderReranker(Reranker):
    """Simple reranking fallback based on lexical overlap with the query."""

    def rerank(self, query: str, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        """Boost results whose content overlaps strongly with the query."""
        query_tokens = {token.lower() for token in query.split() if token}
        if not query_tokens:
            return documents

        def score(document: Document | Chunk) -> float:
            tokens = {token.lower() for token in document.text.split() if token}
            return len(query_tokens & tokens) / max(len(query_tokens), 1)

        for result in documents:
            result.score = float(result.score) + score(result.document)
        return sorted(documents, key=lambda item: item.score, reverse=True)
