"""Unit tests for vector stores."""

from __future__ import annotations

from pygenai.core.data import Document
from pygenai.vectorstore import InMemoryVectorStore


def test_in_memory_vector_store_similarity_search() -> None:
    store = InMemoryVectorStore()
    docs = [
        Document(id="a", text="The quick brown fox jumps over the lazy dog.", metadata={"section": "animals"}),
        Document(id="b", text="Python is a programming language used by data scientists.", metadata={"section": "tech"}),
    ]

    store.add_documents(docs)
    results = store.similarity_search("lazy dog", k=1)

    assert results[0].document.id == "a"
    assert results[0].score >= 0.0


def test_in_memory_vector_store_delete_removes_items() -> None:
    store = InMemoryVectorStore()
    store.add_documents([Document(id="hello", text="Delete me", metadata={})])

    store.delete(["hello"])

    assert store.similarity_search("Delete me", k=5) == []
