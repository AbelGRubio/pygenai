"""Tests for the core RAG abstractions."""

from pathlib import Path

from pygenai.rag import FixedSizeChunker, HybridRetriever, TextLoader
from pygenai.vectorstore import InMemoryVectorStore
from pygenai.core.data import Document


def test_text_loader_creates_document(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("Alpha beta gamma\nDelta epsilon", encoding="utf-8")

    loader = TextLoader()
    documents = loader.load(source)

    assert len(documents) == 1
    assert documents[0].text == "Alpha beta gamma\nDelta epsilon"
    assert documents[0].metadata["path"] == str(source)


def test_fixed_size_chunker_splits_documents() -> None:
    document = Document(
        id="doc-1",
        text="alpha beta gamma delta epsilon zeta eta theta iota kappa",
        metadata={"source": "demo"},
    )

    chunks = FixedSizeChunker(chunk_size=20, overlap=5).chunk_document(document)

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert chunks[0].document_id == "doc-1"


def test_in_memory_store_and_hybrid_retriever() -> None:
    store = InMemoryVectorStore()
    docs = [
        Document(id="a", text="The quick brown fox jumps over the lazy dog.", metadata={"section": "animals"}),
        Document(id="b", text="Python is a programming language used by data scientists.", metadata={"section": "tech"}),
    ]
    store.add_documents(docs)

    results = store.similarity_search("lazy dog", k=1)
    assert results[0].document.id == "a"

    retriever = HybridRetriever(store)
    ranked = retriever.retrieve("Python language", top_k=1)
    assert ranked[0].document.id == "b"
