"""Tests for vectorstore adapter implementations.

====================================================================================================
Name:         libs/vectorstore/tests/vectorstore/test_adapters.py
Description:  Covers mock behavior for in-memory, FAISS, Chroma, Pinecone, and Qdrant stores.
Project:      Pygenai
Date:         2026-11-25
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from pygenai.core.data import Document
from pygenai.rag.embeddings import SimpleEmbeddings
from pygenai.vectorstore import chroma, faiss, in_memory, pinecone, qdrant


class TestInMemoryAdapter:
    """Tests for the LangChain-backed in-memory adapter."""

    def test_create_openai_embeddings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Construct the LiteLLM embedding adapter with supplied settings."""
        captured: dict[str, Any] = {}

        def fake_embeddings(**kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(in_memory, "LiteLLMEmbeddings", fake_embeddings)

        result = in_memory.create_openai_embeddings(api_key="key", model="model", base_url="url")

        assert captured == {"api_key": "key", "model": "model", "base_url": "url"}
        assert isinstance(result, object)

    def test_optional_openai_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Route API settings to the OpenAI-compatible embedding factory."""
        created: list[dict[str, Any]] = []

        def fake_create_openai_embeddings(**kwargs: Any) -> object:
            created.append(kwargs)
            return object()

        monkeypatch.setattr(in_memory, "create_openai_embeddings", fake_create_openai_embeddings)
        monkeypatch.setattr(in_memory, "LangChainInMemoryVectorStore", lambda embedding: SimpleNamespace())

        in_memory.InMemoryVectorStore(openai_api_key="key", openai_model="model", openai_base_url="url")

        assert created == [{"api_key": "key", "model": "model", "base_url": "url"}]

    def test_add_search_filter_delete_and_langchain_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercise document conversion, memory filtering, search, and deletion."""
        calls: dict[str, Any] = {}
        lc_document = SimpleNamespace(page_content="A", metadata={"memory_type": "short_term"}, id="doc-1")

        class FakeStore:
            def add_documents(self, documents: list[Any], ids: list[str]) -> list[str]:
                calls["add"] = (documents, ids)
                return ids

            def similarity_search_with_score(self, query: str, k: int, filter: Any) -> list[Any]:
                calls["search"] = (query, k, filter)
                return [(lc_document, 0.25)]

            def delete(self, ids: list[str]) -> None:
                calls["delete"] = ids

        monkeypatch.setattr(in_memory, "LangChainInMemoryVectorStore", lambda embedding: FakeStore())
        monkeypatch.setattr(in_memory, "to_langchain_document", lambda document: lc_document)
        monkeypatch.setattr(in_memory, "from_langchain_document", lambda document: Document(id="doc-1", text="A", metadata=document.metadata))

        store = in_memory.InMemoryVectorStore(embeddings=SimpleEmbeddings())
        ids = store.add_documents([Document(id="doc-1", text="A")])
        assert ids == ["doc-1"]

        store_result = store.similarity_search("A", k=2, filter_search={"memory_type": "short_term", "user_id": None})
        assert store_result[0].score == 0.25
        assert calls["search"][2](lc_document) is True

        store.delete(["doc-1"])
        assert calls["delete"] == ["doc-1"]


class TestFAISSAdapter:
    """Tests for the FAISS adapter."""

    def test_missing_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raise a clear import error when FAISS is unavailable."""
        monkeypatch.setattr(faiss, "faiss", None)
        with pytest.raises(ImportError, match="FAISS support"):
            faiss.FAISSVectorStore()

    def test_l2_add_search_filter_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercise L2 indexing, metadata filtering, and deletion rebuilding."""
        class FakeIndex:
            def __init__(self, dimension: int) -> None:
                self.dimension = dimension
                self.vectors: list[Any] = []

            def add(self, matrix: Any) -> None:
                self.vectors.append(matrix)

            def search(self, vector: Any, k: int) -> tuple[Any, Any]:
                del vector
                return [[0.5, 99.0]], [[0, -1][:k]]

        fake_module = SimpleNamespace(IndexFlatL2=lambda dimension: FakeIndex(dimension), IndexFlatIP=lambda dimension: FakeIndex(dimension))
        monkeypatch.setattr(faiss, "faiss", fake_module)
        monkeypatch.setattr(faiss, "numpy", None, raising=False)

        store = faiss.FAISSVectorStore(dimension=2)
        ids = store.add_documents([Document(id="doc-1", text="A", metadata={"memory_type": "long_term"})], embeddings=[[0.1, 0.2]])
        assert ids == ["doc-1"]

        hits = store.similarity_search("ignored", k=1, filter_search={"query_vector": [0.1, 0.2], "memory_type": "long_term"})
        assert len(hits) == 1
        assert hits[0].document.id == "doc-1"

        store.delete(["doc-1"])
        assert store._documents == []

        ip_store = faiss.FAISSVectorStore(dimension=2, metric="ip")
        ip_store.add_documents([Document(id="doc-2", text="B")], embeddings=[[0.3, 0.4]])
        ip_hits = ip_store.similarity_search("ignored", k=1, filter_search={"query_vector": [0.3, 0.4]})
        assert ip_hits[0].document.id == "doc-2"

    def test_add_and_search_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reject missing embeddings, mismatched lengths, and missing query vectors."""
        fake_module = SimpleNamespace(IndexFlatL2=lambda dimension: SimpleNamespace(), IndexFlatIP=lambda dimension: SimpleNamespace())
        monkeypatch.setattr(faiss, "faiss", fake_module)
        store = faiss.FAISSVectorStore()

        with pytest.raises(ValueError, match="explicit embeddings"):
            store.add_documents([Document(id="1", text="A")])
        with pytest.raises(ValueError, match="same length"):
            store.add_documents([Document(id="1", text="A")], embeddings=[])
        with pytest.raises(ValueError, match="query vector"):
            store.similarity_search("query")


class TestChromaAdapter:
    """Tests for the ChromaDB adapter."""

    def test_missing_dependency_and_persistent_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cover dependency validation and persistent client setup."""
        monkeypatch.setattr(chroma, "chromadb", None)
        with pytest.raises(ImportError, match="Chroma support"):
            chroma.ChromaVectorStore()

        calls: dict[str, Any] = {}
        collection = SimpleNamespace(add=lambda **kwargs: calls.update(kwargs), query=lambda **kwargs: {"documents": [[]]}, delete=lambda **kwargs: calls.update(kwargs))
        class FakeClient:
            def get_or_create_collection(self, name: str) -> Any:
                calls["name"] = name
                return collection
        monkeypatch.setattr(chroma, "chromadb", SimpleNamespace(PersistentClient=lambda path: FakeClient(), Client=lambda: FakeClient()))

        store = chroma.ChromaVectorStore(persist_directory="/tmp/chroma")
        ids = store.add_documents([Document(id="1", text="A", metadata={"x": 1})], embeddings=[[0.1]])
        assert ids == ["1"]
        assert calls["embeddings"] == [[0.1]]
        assert calls["name"] == "pygenai"
        assert store.similarity_search("A") == []
        store.delete(["1"])

    def test_query_with_filter_and_empty_optional_arrays(self) -> None:
        """Build metadata filters and handle sparse Chroma response arrays."""
        collection = SimpleNamespace(
            add=lambda **kwargs: None,
            query=lambda **kwargs: {"documents": [["A", "B"]], "metadatas": [[None, {"topic": "b"}]], "distances": [[1.0]], "ids": [[]]},
            delete=lambda **kwargs: None,
        )
        client = SimpleNamespace(get_or_create_collection=lambda name: collection)
        store = chroma.ChromaVectorStore(client=client)

        hits = store.similarity_search("A", k=2, filter_search={"topic": "a"})

        assert len(hits) == 2
        assert hits[0].document.id == "chroma-hit-0"
        assert hits[1].document.metadata == {"topic": "b"}


class TestPineconeAdapter:
    """Tests for the Pinecone adapter."""

    def test_missing_dependency_and_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cover missing dependency and input validation."""
        monkeypatch.setattr(pinecone, "Pinecone", None)
        with pytest.raises(ImportError, match="Pinecone support"):
            pinecone.PineconeVectorStore("index")

        index = SimpleNamespace(upsert=lambda **kwargs: None, query=lambda **kwargs: {"matches": []}, delete=lambda **kwargs: None)
        store = pinecone.PineconeVectorStore("index", client=SimpleNamespace(Index=lambda name: index), namespace="ns")
        with pytest.raises(ValueError, match="explicit embeddings"):
            store.add_documents([Document(id="1", text="A")])
        with pytest.raises(ValueError, match="same length"):
            store.add_documents([Document(id="1", text="A")], embeddings=[])
        with pytest.raises(ValueError, match="query vector"):
            store.similarity_search("query")

    def test_upsert_query_filters_and_delete(self) -> None:
        """Exercise Pinecone upsert, top-level and nested filters, and delete."""
        calls: dict[str, Any] = {}
        index = SimpleNamespace(
            upsert=lambda **kwargs: calls.update({"upsert": kwargs}),
            query=lambda **kwargs: calls.update({"query": kwargs}) or {"matches": [{"id": "1", "score": 0.8, "metadata": {"text": "A", "topic": "x"}}]},
            delete=lambda **kwargs: calls.update({"delete": kwargs}),
        )
        store = pinecone.PineconeVectorStore("index", client=SimpleNamespace(Index=lambda name: index), namespace="ns")

        ids = store.add_documents([Document(id="1", text="A", metadata={"topic": "x"})], embeddings=[[0.1, 0.2]])
        hits = store.similarity_search("ignored", k=2, filter_search={"query_vector": [0.1, 0.2], "filter": {"topic": "x"}, "memory_type": "long_term"})
        store.delete(["1"])

        assert ids == ["1"]
        assert hits[0].document.text == "A"
        assert calls["query"]["filter"] == {"topic": "x", "memory_type": "long_term"}
        assert calls["delete"] == {"ids": ["1"], "namespace": "ns"}


class TestQdrantAdapter:
    """Tests for the Qdrant adapter."""

    def test_missing_dependency_and_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cover missing dependency and input validation."""
        monkeypatch.setattr(qdrant, "QdrantClient", None)
        monkeypatch.setattr(qdrant, "models", None)
        with pytest.raises(ImportError, match="Qdrant support"):
            qdrant.QdrantVectorStore("collection")

        class EmptyModels:
            pass
        monkeypatch.setattr(qdrant, "models", EmptyModels)
        client = SimpleNamespace(upsert=lambda **kwargs: None, search=lambda **kwargs: [], delete=lambda **kwargs: None)
        store = qdrant.QdrantVectorStore("collection", client=client)
        with pytest.raises(ValueError, match="explicit embeddings"):
            store.add_documents([Document(id="1", text="A")])
        with pytest.raises(ValueError, match="same length"):
            store.add_documents([Document(id="1", text="A")], embeddings=[])
        with pytest.raises(ValueError, match="query vector"):
            store.similarity_search("query")

    def test_upsert_search_filter_and_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercise Qdrant point creation, search, payload conversion, and delete."""
        calls: dict[str, Any] = {}

        class ModelFactory:
            class PointStruct:
                def __init__(self, **kwargs: Any) -> None:
                    self.__dict__.update(kwargs)

            class FilterSelector:
                def __init__(self, **kwargs: Any) -> None:
                    self.__dict__.update(kwargs)

            class Filter:
                def __init__(self, **kwargs: Any) -> None:
                    self.__dict__.update(kwargs)

            class FieldCondition:
                def __init__(self, **kwargs: Any) -> None:
                    self.__dict__.update(kwargs)

            class MatchAny:
                def __init__(self, **kwargs: Any) -> None:
                    self.__dict__.update(kwargs)

        monkeypatch.setattr(qdrant, "models", ModelFactory)
        client = SimpleNamespace(
            upsert=lambda **kwargs: calls.update({"upsert": kwargs}),
            search=lambda **kwargs: calls.update({"search": kwargs}) or [SimpleNamespace(payload={"text": "A", "original_id": "1", "topic": "x"}, id=7, score=0.75)],
            delete=lambda **kwargs: calls.update({"delete": kwargs}),
        )
        store = qdrant.QdrantVectorStore("collection", client=client)

        ids = store.add_documents([Document(id="1", text="A", metadata={"topic": "x"})], embeddings=[[0.1, 0.2]])
        hits = store.similarity_search("ignored", k=3, filter_search={"query_vector": [0.1, 0.2], "filter": {"topic": "x"}})
        store.delete(["1"])

        assert ids == ["1"]
        assert hits[0].document.text == "A"
        assert hits[0].score == 0.75
        assert calls["search"]["query_filter"] == {"topic": "x"}
        assert calls["delete"]["collection_name"] == "collection"
