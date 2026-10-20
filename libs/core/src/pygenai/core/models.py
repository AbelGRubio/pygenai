"""Core data models for the memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class MessageRole(StrEnum):
    """Role of the message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseMessage):
    """Represents a single message in a conversation.

    Attributes:
        id: Unique identifier for the message.
        role: Role of the sender (user, assistant, system).
        content: The message text content.
        tokens: Number of tokens in the message (estimated via tiktoken).
        metadata: Additional metadata (e.g., source, model, confidence).
        timestamp: When the message was created.
    """

    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "token": self.tokens,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SummaryRecord:
        """Create summary from dictionary."""
        return cls(
            tokens=data.get("tokens", 0),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class SummaryRecord(BaseModel):
    """Represents a compressed summary of past conversation segments.

    Attributes:
        id: Unique identifier for the summary.
        messages_count: Number of original messages summarized.
        original_tokens: Total tokens in the summarized messages.
        summary: The condensed summary text.
        metadata: Original context (e.g., date range, topics).
        timestamp: When the summary was created.
    """

    summary: str
    messages_count: int = 0
    original_tokens: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary (compatible wrapper for Pydantic v2)."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SummaryRecord:
        """Create summary from dictionary (compatible wrapper for Pydantic v2)."""
        return cls.model_validate(data)


class ConversationState(BaseModel):
    """Represents the complete state of a conversation.

    Attributes:
        session_id: Unique session identifier.
        messages: Current active messages in the conversation.
        summaries: Historical summaries of past conversation segments.
        total_tokens: Total tokens across all messages.
        metadata: Session-level metadata.
    """

    session_id: str
    messages: list[Message] = field(default_factory=list)
    summaries: list[SummaryRecord] = field(default_factory=list)
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary (compatible wrapper for Pydantic v2)."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SummaryRecord:
        """Create summary from dictionary (compatible wrapper for Pydantic v2)."""
        return cls.model_validate(data)


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
