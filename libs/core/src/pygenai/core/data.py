"""Shared data contracts for the RAG package.

========================================================================================================================
Name:         pygenai/rag/data.py
Description:  Shared document, chunk, and retrieval data models used across loaders, chunkers, vector stores,
              and retrievers.
Project:      Pygenai
Date:         2026-10-02 12:19:24
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document as LangChainDocument


@dataclass(slots=True)
class Document:
    """Canonical text unit stored in a RAG pipeline.

    This model is intentionally lightweight so it can be used by both raw file
    loaders and chunked retrieval systems without forcing a database-specific
    schema.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def as_chunk(self, *, index: int = 0, start: int = 0, end: int | None = None) -> Chunk:
        """Convert the document into a chunk with the same content.

        Args:
            index: Sequential chunk index.
            start: Character offset within the original source text.
            end: Optional end offset within the original source text.

        Returns:
            Chunk: A chunk representation derived from this document.
        """
        return Chunk(
            id=f"{self.id}-chunk-{index}",
            text=self.text,
            metadata={**self.metadata, "source": self.source},
            document_id=self.id,
            chunk_index=index,
            start=start,
            end=end,
        )


@dataclass(slots=True)
class Chunk:
    """A chunk derived from a document after segmentation."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    document_id: str | None = None
    chunk_index: int = 0
    start: int = 0
    end: int | None = None

    def as_document(self) -> Document:
        """Return a document-shaped copy of the chunk."""
        return Document(
            id=self.document_id or self.id,
            text=self.text,
            metadata={**self.metadata},
            source=self.metadata.get("source"),
        )


@dataclass(slots=True)
class SearchHit:
    """A retrieval result with a minimal score value."""

    document: Document | Chunk
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def to_langchain_document(value: Document | Chunk | LangChainDocument) -> LangChainDocument:
    """Convert a document-like object into a langchain_core Document."""
    metadata = dict(getattr(value, "metadata", {}) or {})
    text = getattr(value, "text", None)
    if text is None:
        text = getattr(value, "page_content", "")

    document_id = getattr(value, "document_id", None)
    item_id = getattr(value, "id", None)
    if document_id is not None:
        metadata.setdefault("document_id", document_id)
    if item_id is not None:
        metadata.setdefault("doc_id", item_id)
    if getattr(value, "source", None) is not None:
        metadata.setdefault("source", value.source)

    langchain_document = LangChainDocument(page_content=text, metadata=metadata)
    if item_id is not None:
        langchain_document.id = str(item_id)
    return langchain_document


def from_langchain_document(value: LangChainDocument) -> Document:
    """Convert a langchain_core Document to the package's typed model."""
    document_id = (
        getattr(value, "id", None)
        or value.metadata.get("document_id")
        or value.metadata.get("doc_id")
        or "langchain-doc"
    )
    return Document(
        id=str(document_id),
        text=value.page_content,
        metadata=dict(value.metadata),
        source=value.metadata.get("source"),
    )


__all__ = ["Chunk", "Document", "SearchHit", "from_langchain_document", "to_langchain_document"]
