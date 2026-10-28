"""In-memory vector store for local development, tests, and agent memory.

========================================================================================================================
Name:         pygenai/vectorstores/in_memory.py
Description:  Provides a lightweight vector store backed by LangChain and Python lists, supporting RAG and agent memory.
Project:      Pygenai
Date:         2026-10-22 17:35:00
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore as LangChainInMemoryVectorStore

from pygenai.core.data import Chunk, Document, SearchHit, from_langchain_document, to_langchain_document
from pygenai.rag.embeddings import LiteLLMEmbeddings, SimpleEmbeddings

from .base import VectorStore


def create_openai_embeddings(
    *,
    api_key: str | None = None,
    model: str = "gemini/gemini-embedding-2",
    base_url: str | None = None,
) -> Embeddings:
    """Create an OpenAI-backed embeddings client when the optional dependency is installed."""
    return LiteLLMEmbeddings(api_key=api_key, model=model, base_url=base_url)


class InMemoryVectorStore(VectorStore):
    """Local vector store facade built on top of LangChain's in-memory implementation for RAG and memory."""

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        *,
        openai_api_key: str | None = None,
        openai_model: str = "gemini/gemini-embedding-2",
        openai_base_url: str | None = None,
    ) -> None:
        """Initialize the LangChain-backed in-memory store.

        Args:
            embeddings: Explicit embeddings instance. If not provided, a lightweight
                deterministic fallback is used unless OpenAI settings are supplied.
            openai_api_key: Optional API key for OpenAI embeddings.
            openai_model: Model name to use when constructing OpenAI embeddings.
            openai_base_url: Optional custom OpenAI-compatible endpoint.
        """
        if embeddings is None and (openai_api_key is not None or openai_base_url is not None):
            embeddings = create_openai_embeddings(
                api_key=openai_api_key,
                model=openai_model,
                base_url=openai_base_url,
            )
        self._embedding_model = embeddings or SimpleEmbeddings()
        self._store = LangChainInMemoryVectorStore(embedding=self._embedding_model)
        self._filter_search = {}
        self._memory_keys = ["memory_type", "user_id", "session_id"]

    def add_documents(
        self,
        documents: Sequence[Document | Chunk],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Add documents and return their generated identifiers."""
        del embeddings
        langchain_documents = [to_langchain_document(document) for document in documents]
        document_ids = [str(getattr(document, "id", "")) for document in documents]
        return self._store.add_documents(langchain_documents, ids=document_ids)

    def _langchain_filter(self, doc: Document) -> Any:
        """Create a LangChain-compatible filter function for memory attributes."""
        return all(
            doc.metadata.get(k_sub) == v_sub
            for k_sub, v_sub in self._filter_search.items()
            if k_sub in self._memory_keys and v_sub is not None
        )

    def similarity_search(
        self, query: str, k: int = 5, *, filter_search: dict[str, Any] | None = None
    ) -> list[SearchHit]:
        """Return the best matches with their scores using LangChain vector search and metadata filters."""
        # Normalize filter parameters for memory/RAG attributes if passed via flat dictionary
        langchain_filter = filter_search
        if filter_search and "filter" not in filter_search and any(key in filter_search for key in self._memory_keys):
            # Check if we have top-level memory keys like memory_type, user_id, session_id
            self._filter_search = filter_search
            langchain_filter = self._langchain_filter

        matches = self._store.similarity_search_with_score(query, k=k, filter=langchain_filter)
        results: list[SearchHit] = []
        for document, score in matches:
            results.append(
                SearchHit(
                    document=from_langchain_document(document),
                    score=float(score),
                    metadata=dict(document.metadata),
                )
            )
        return results

    def delete(self, ids: Sequence[str]) -> None:
        """Delete stored entries using LangChain's underlying store."""
        self._store.delete(ids=list(ids))


__all__ = ["InMemoryVectorStore", "create_openai_embeddings"]
