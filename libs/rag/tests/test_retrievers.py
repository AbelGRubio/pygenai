"""Unit tests for retrievers and rerankers."""

from __future__ import annotations

from pygenai.core.data import Document
from pygenai.rag.retrievers import CrossEncoderReranker, HybridRetriever, NoOpReranker, VectorRetriever
from pygenai.vectorstore import InMemoryVectorStore


def test_vector_retriever_returns_top_matches() -> None:
    store = InMemoryVectorStore()
    store.add_documents([
        Document(id="a", text="Python is a programming language.", metadata={}),
        Document(id="b", text="Cats are independent pets.", metadata={}),
    ])

    retriever = VectorRetriever(store)
    results = retriever.retrieve("Python", top_k=1)

    assert results[0].document.id == "a"


def test_hybrid_retriever_blends_keyword_and_vector_scores() -> None:
    store = InMemoryVectorStore()
    store.add_documents([
        Document(id="a", text="Machine learning with Python.", metadata={}),
        Document(id="b", text="Cats are great companions.", metadata={}),
    ])

    retriever = HybridRetriever(store)
    results = retriever.retrieve("Python", top_k=1)

    assert results[0].document.id == "a"


def test_noop_reranker_keeps_order() -> None:
    reranker = NoOpReranker()
    documents = [
        type("Result", (), {"score": 0.5, "document": Document(id="x", text="foo", metadata={})})(),
        type("Result", (), {"score": 0.9, "document": Document(id="y", text="bar", metadata={})})(),
    ]

    result = reranker.rerank("query", documents)

    assert [item.document.id for item in result] == ["x", "y"]


def test_cross_encoder_reranker_updates_order() -> None:
    reranker = CrossEncoderReranker()
    documents = [
        type("Result", (), {"score": 0.2, "document": Document(id="x", text="dogs bark", metadata={})})(),
        type("Result", (), {"score": 0.1, "document": Document(id="y", text="python data science", metadata={})})(),
    ]

    result = reranker.rerank("python", documents)

    assert result[0].document.id == "y"
