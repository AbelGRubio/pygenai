"""Example: semantic and hybrid retrieval against a small local index."""

from __future__ import annotations

from pygenai.core.data import Document
from pygenai.rag.retrievers import HybridRetriever, VectorRetriever
from pygenai.rag.vectorstores import InMemoryVectorStore


def main() -> None:
    """Run semantic and hybrid retrieval over a tiny dataset."""
    store = InMemoryVectorStore()
    store.add_documents([
        Document(
            id="p1", text="Python is used for data analysis and machine learning.", metadata={"category": "science"}
        ),
        Document(id="p2", text="Cats are known for being independent pets.", metadata={"category": "animals"}),
    ])

    semantic = VectorRetriever(store)
    hybrid = HybridRetriever(store)

    print("Semantic:", semantic.retrieve("Python", top_k=1)[0].document.text)
    print("Hybrid:", hybrid.retrieve("Python", top_k=1)[0].document.text)


if __name__ == "__main__":
    main()
