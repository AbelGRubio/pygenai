"""High-level agent memory manager utilizing vector store backends.

========================================================================================================================
Name:         pygenai/vectorstores/memory.py
Description:  Unified abstraction to manage short-term context and long-term semantic agent memories.
Project:      Pygenai
Date:         2026-10-22 17:16:00
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Literal
import uuid

from pygenai.core.data import SearchHit
from .base import MemoryRecord, VectorStore


class AgentMemoryManager:
    """Manages short-term and long-term memory operations over a VectorStore."""

    def __init__(self, vector_store: VectorStore) -> None:
        """Initialize memory manager with a target vector store instance."""
        self.store = vector_store

    def save_memory(
        self,
        content: str,
        *,
        memory_type: Literal["short_term", "long_term"] = "long_term",
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: Sequence[float] | None = None,
    ) -> str:
        """Store a new memory item into short or long term storage."""
        record_id = f"mem-{uuid.uuid4().hex[:12]}"
        record = MemoryRecord(
            id=record_id,
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        embeddings = [embedding] if embedding is not None else None
        self.store.add_memories([record], embeddings=embeddings)
        return record_id

    def recall(
        self,
        query: str,
        k: int = 5,
        *,
        memory_type: Literal["short_term", "long_term"] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        extra_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Recall relevant memories matching query context and filters."""
        filter_search: dict[str, Any] = extra_filter or {}
        if memory_type:
            filter_search["memory_type"] = memory_type
        if user_id:
            filter_search["user_id"] = user_id
        if session_id:
            filter_search["session_id"] = session_id

        return self.store.similarity_search(query=query, k=k, filter_search=filter_search)

    def clear_short_term_session(self, session_id: str) -> None:
        """Remove short-term memories belonging to a specific session."""
        recent_memories = self.recall("", k=100, memory_type="short_term", session_id=session_id)
        ids_to_delete = [hit.document.id for hit in recent_memories if hit.document.id]
        if ids_to_delete:
            self.store.delete(ids_to_delete)


__all__ = ["AgentMemoryManager"]