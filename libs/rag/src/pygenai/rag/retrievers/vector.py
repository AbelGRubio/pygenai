"""Semantic vector retriever.

========================================================================================================================
Name:         pygenai/rag/retrievers/vector.py
Description:  Retrieves the most relevant chunks using vector similarity.
Project:      Pygenai
Date:         2026-10-02 12:29:54
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from pygenai.core.base_retriever import BaseRetriever, RetrievedDocument
from pygenai.vectorstore.base import VectorStore


class VectorRetriever(BaseRetriever):
    """Simple semantic retriever backed by a vector store."""

    def __init__(self, vector_store: VectorStore) -> None:
        """Store the target vector store used for semantic search."""
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Query the vector store and package the results as retrieval hits."""
        hits = self.vector_store.similarity_search(query, k=top_k)
        results: list[RetrievedDocument] = []
        for rank, hit in enumerate(hits, start=1):
            results.append(
                RetrievedDocument(
                    document=hit.document,
                    score=float(hit.score),
                    rank=rank,
                    metadata={**hit.metadata},
                )
            )
        return results
