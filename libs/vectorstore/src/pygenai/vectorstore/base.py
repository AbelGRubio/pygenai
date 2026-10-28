"""Base abstractions for vector databases and agent memory storage.

========================================================================================================================
Name:         pygenai/vectorstores/base.py
Description:  Defines a common API for document retrieval, semantic search, and multi-tier agent memory.
Project:      Pygenai
Date:         2026-10-22 17:16:00
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pygenai.core.data import Chunk, Document, SearchHit


@dataclass
class MemoryRecord:
    """Represents a discrete unit of memory stored in a vector store.

    Attributes:
        id: Unique identifier for the memory record.
        content: Textual observation, summary, or user interaction.
        memory_type: Indicates lifetime scope ('short_term' or 'long_term').
        user_id: Optional user identifier for user-level isolation.
        session_id: Optional session identifier for conversation tracing.
        tags: Categorical tags attached to the record.
        timestamp: Time when the memory was recorded.
        metadata: Additional arbitrary contextual key-value pairs.
    """

    id: str
    content: str
    memory_type: Literal["short_term", "long_term"] = "long_term"
    user_id: str | None = None
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> Document:
        """Convert memory record into a standardized Document object."""
        merged_meta = {
            **self.metadata,
            "memory_type": self.memory_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat(),
            "is_memory": True,
        }
        return Document(id=self.id, text=self.content, metadata=merged_meta)

    @classmethod
    def from_document(cls, doc: Document) -> MemoryRecord:
        """Reconstruct a MemoryRecord from a Document object."""
        meta = dict(doc.metadata or {})
        ts_str = meta.pop("timestamp", None)
        timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.now(UTC)

        return cls(
            id=str(doc.id),
            content=doc.text,
            memory_type=meta.pop("memory_type", "long_term"),
            user_id=meta.pop("user_id", None),
            session_id=meta.pop("session_id", None),
            tags=meta.pop("tags", []),
            timestamp=timestamp,
            metadata=meta,
        )


class VectorStore(ABC):
    """Abstract interface used by retrievers and memory controllers."""

    @abstractmethod
    def add_documents(
        self,
        documents: Sequence[Document | Chunk],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Add a collection of documents/chunks and optional precomputed embeddings."""

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        *,
        filter_search: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return the top-k matching documents or memories for a query."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Remove entries by unique identifiers."""

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Fetch documents by their explicit IDs. Override in subclasses for optimization."""
        raise NotImplementedError("Direct ID retrieval is not implemented for this store.")

    def add_memories(
        self,
        records: Sequence[MemoryRecord],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Convenience method to store agent memory records."""
        docs = [record.to_document() for record in records]
        return self.add_documents(docs, embeddings=embeddings)


__all__ = ["MemoryRecord", "VectorStore"]
