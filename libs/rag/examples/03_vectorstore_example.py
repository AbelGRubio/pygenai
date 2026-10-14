"""Example: storing and searching documents using LangChain's in-memory vector store."""

from __future__ import annotations

from langchain_core.documents import Document as LangChainDocument

from pygenai.rag.vectorstores import InMemoryVectorStore


def main() -> None:
    """Add a few documents and perform a similarity query."""
    store = InMemoryVectorStore()
    documents = [
        LangChainDocument(page_content="The quick brown fox jumps over the lazy dog.", metadata={"section": "animals"}),
        LangChainDocument(
            page_content="Python is a programming language used in AI and data work.", metadata={"section": "tech"}
        ),
    ]

    store.add_documents(documents)
    results = store.similarity_search("lazy dog", k=1)

    print(results[0].document.text)
    print(results[0].score)


if __name__ == "__main__":
    main()
