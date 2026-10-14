"""Example: splitting a document with different chunking strategies."""

from __future__ import annotations

from pygenai.core.data import Document
from pygenai.rag.chunking import FixedSizeChunker, TokenChunker


def main() -> None:
    """Create a document and segment it with a couple of strategies."""
    document = Document(
        id="demo-doc",
        text="Python helps you build fast prototypes. It also supports RAG pipelines, embeddings, and retrieval workflows.",
        metadata={"source": "example"},
    )

    fixed_chunks = FixedSizeChunker(chunk_size=30, overlap=5).chunk_document(document)
    token_chunks = TokenChunker(max_tokens=5, overlap_tokens=1).chunk_document(document)

    print("Fixed size chunks:", len(fixed_chunks))
    print("Token chunks:", len(token_chunks))
    print("First token chunk:", token_chunks[0].text)


if __name__ == "__main__":
    main()
