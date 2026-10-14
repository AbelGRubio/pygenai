"""End-to-end RAG pipeline example using loaders, chunking, vector store, and retrievers.

This script simulates how a real LLM application would work:
1. load a source document from disk,
2. split it into chunks,
3. index those chunks in a vector store,
4. retrieve the most relevant chunks for a user question,
5. build a context prompt for a model.
"""

from __future__ import annotations

from pathlib import Path

from pygenai.rag.chunking import FixedSizeChunker
from pygenai.rag.loaders import TextLoader
from pygenai.rag.retrievers import HybridRetriever
from pygenai.rag.vectorstores import InMemoryVectorStore


def build_demo_document() -> Path:
    """Create a small knowledge base file for the example."""
    path = Path("./examples/example_documents/rag_demo.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
Pygenai is a Python toolkit for building AI assistants and retrieval pipelines.

The project includes document loaders for text, markdown, CSV, PDF, and web sources.
It also supports chunking strategies such as fixed-size, token-based, semantic, and structural segmentation.

For retrieval, the system can search with vector similarity, hybrid ranking, and rerankers.
This makes it useful for building enterprise RAG systems with grounding from local documents.

A typical workflow starts by loading raw files, splitting them into chunks, storing embeddings,
then retrieving the most relevant context to answer a user's question.
""".strip(),
        encoding="utf-8",
    )
    return path


def build_rag_pipeline() -> tuple[InMemoryVectorStore, HybridRetriever]:
    """Load, chunk, and index the demo document."""
    source = build_demo_document()
    loaded_documents = TextLoader().load(source)
    chunker = FixedSizeChunker(chunk_size=220, overlap=40)

    chunks = []
    for document in loaded_documents:
        chunks.extend(chunker.chunk_document(document))

    store = InMemoryVectorStore()
    store.add_documents(chunks)

    retriever = HybridRetriever(store)
    return store, retriever


def build_llm_prompt(question: str, context: list[str]) -> str:
    """Simulate the prompt sent to an LLM with retrieved context."""
    merged_context = "\n\n".join(context)
    return (
        "You are a helpful assistant. Use only the context below to answer the user question.\n\n"
        f"Context:\n{merged_context}\n\n"
        f"Question: {question}\n\n"
        "Answer briefly and accurately."
    )


def main() -> None:
    """Run the end-to-end example and print the generated LLM prompt."""
    _, retriever = build_rag_pipeline()

    question = "¿Qué tipos de loaders y estrategias de chunking admite Pygenai?"
    results = retriever.retrieve(question, top_k=3)

    context_chunks = [result.document.text for result in results]
    prompt = build_llm_prompt(question, context_chunks)

    print("=== Query ===")
    print(question)
    print("\n=== Retrieved context ===")
    for index, chunk in enumerate(results, start=1):
        print(f"[{index}] {chunk.document.text}\n")

    print("\n=== Simulated LLM prompt ===")
    print(prompt)


if __name__ == "__main__":
    main()
