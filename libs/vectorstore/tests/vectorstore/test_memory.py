"""Tests for agent memory management.

====================================================================================================
Name:         libs/vectorstore/tests/vectorstore/test_memory.py
Description:  Covers memory creation, recall filters, and session cleanup.
Project:      Pygenai
Date:         2026-11-25
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pygenai.core.data import Chunk, Document, SearchHit
from pygenai.vectorstore.base import MemoryRecord, VectorStore
from pygenai.vectorstore.memory import AgentMemoryManager


class RecordingVectorStore(VectorStore):
    """Vector store that records memory operations for assertions."""

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []
        self.embeddings: Sequence[Sequence[float]] | None = None
        self.search_filter: dict[str, Any] | None = None
        self.deleted_ids: list[str] = []

    def add_documents(self, documents: Sequence[Document | Chunk], *, embeddings: Sequence[Sequence[float]] | None = None) -> list[str]:
        """Record documents as memory records through add_memories."""
        del documents, embeddings
        return []

    def add_memories(self, records: Sequence[MemoryRecord], *, embeddings: Sequence[Sequence[float]] | None = None) -> list[str]:
        """Record memory values and optional embeddings."""
        self.records.extend(records)
        self.embeddings = embeddings
        return [record.id for record in records]

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        *,
        filter_search: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return a hit and capture the recall filter."""
        del query, k
        self.search_filter = filter_search
        return [SearchHit(document=Document(id="mem-hit", text="result"), score=0.9)]

    def delete(self, ids: Sequence[str]) -> None:
        """Capture identifiers selected for deletion."""
        self.deleted_ids.extend(ids)


class TestAgentMemoryManager:
    """Tests for AgentMemoryManager operations."""

    def test_save_memory_with_embedding_and_fields(self) -> None:
        """Create a complete memory record and preserve its optional embedding."""
        store = RecordingVectorStore()
        manager = AgentMemoryManager(store)

        memory_id = manager.save_memory(
            "Review the plan",
            memory_type="short_term",
            user_id="user-1",
            session_id="session-1",
            tags=["work"],
            metadata={"priority": "high"},
            embedding=[0.1, 0.2],
        )

        assert memory_id.startswith("mem-")
        assert store.records[0].id == memory_id
        assert store.records[0].tags == ["work"]
        assert store.records[0].metadata == {"priority": "high"}
        assert store.embeddings == [[0.1, 0.2]]

    def test_save_memory_uses_default_values(self) -> None:
        """Use empty tags and metadata when optional values are omitted."""
        store = RecordingVectorStore()
        manager = AgentMemoryManager(store)

        manager.save_memory("Simple memory")

        assert store.records[0].tags == []
        assert store.records[0].metadata == {}
        assert store.embeddings is None

    def test_recall_combines_all_filters(self) -> None:
        """Merge context and extra filters before querying the store."""
        store = RecordingVectorStore()
        manager = AgentMemoryManager(store)

        hits = manager.recall(
            "plan",
            k=3,
            memory_type="long_term",
            user_id="user-1",
            session_id="session-1",
            extra_filter={"topic": "work"},
        )

        assert hits[0].document.id == "mem-hit"
        assert store.search_filter == {
            "topic": "work",
            "memory_type": "long_term",
            "user_id": "user-1",
            "session_id": "session-1",
        }

    def test_recall_without_optional_filters(self) -> None:
        """Pass an empty filter dictionary when no filters are supplied."""
        store = RecordingVectorStore()
        manager = AgentMemoryManager(store)

        manager.recall("anything")

        assert store.search_filter == {}

    def test_clear_short_term_session_deletes_hits(self) -> None:
        """Delete every returned short-term memory for a session."""
        store = RecordingVectorStore()
        manager = AgentMemoryManager(store)

        manager.clear_short_term_session("session-1")

        assert store.deleted_ids == ["mem-hit"]
        assert store.search_filter is not None
        assert store.search_filter["memory_type"] == "short_term"
        assert store.search_filter["session_id"] == "session-1"

    def test_clear_short_term_session_with_no_hits(self) -> None:
        """Do nothing when recall returns no memories."""
        store = RecordingVectorStore()
        manager = AgentMemoryManager(store)

        class EmptyStore(RecordingVectorStore):
            def similarity_search(
                self,
                query: str,
                k: int = 5,
                *,
                filter_search: dict[str, Any] | None = None,
            ) -> list[SearchHit]:
                del query, k, filter_search
                return []

        empty_store = EmptyStore()
        AgentMemoryManager(empty_store).clear_short_term_session("empty")

        assert empty_store.deleted_ids == []
