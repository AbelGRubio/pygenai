"""Tests for vectorstore base abstractions.

====================================================================================================
Name:         libs/vectorstore/tests/vectorstore/test_base.py
Description:  Covers MemoryRecord conversion and VectorStore default behavior.
Project:      Pygenai
Date:         2026-11-25
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from pygenai.core.data import Chunk, Document, SearchHit
from pygenai.vectorstore.base import MemoryRecord, VectorStore


class ConcreteVectorStore(VectorStore):
    """Minimal concrete vector store used to exercise base helpers."""

    def __init__(self) -> None:
        self.documents: list[Document | Chunk] = []
        self.embeddings: Sequence[Sequence[float]] | None = None

    def add_documents(
        self,
        documents: Sequence[Document | Chunk],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Store documents in memory."""
        self.documents.extend(documents)
        self.embeddings = embeddings
        return [str(document.id) for document in documents]

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        *,
        filter_search: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return no matches for the abstract-store test double."""
        del query, k, filter_search
        return []

    def delete(self, ids: Sequence[str]) -> None:
        """Remove no documents in the test double."""
        del ids


class TestMemoryRecord:
    """Tests for memory record serialization."""

    def test_document_round_trip_with_timestamp(self) -> None:
        """Convert a record to a document and back while preserving metadata."""
        timestamp = datetime(2026, 11, 25, 10, 30, tzinfo=UTC)
        record = MemoryRecord(
            id="memory-1",
            content="Remember this",
            memory_type="short_term",
            user_id="user-1",
            session_id="session-1",
            tags=["important"],
            timestamp=timestamp,
            metadata={"topic": "testing"},
        )

        document = record.to_document()
        restored = MemoryRecord.from_document(document)

        assert document.metadata["is_memory"] is True
        assert restored.id == "memory-1"
        assert restored.content == "Remember this"
        assert restored.memory_type == "short_term"
        assert restored.user_id == "user-1"
        assert restored.session_id == "session-1"
        assert restored.tags == ["important"]
        assert restored.timestamp == timestamp
        assert restored.metadata == {"topic": "testing", "is_memory": True}

    def test_document_round_trip_without_timestamp(self) -> None:
        """Use the default timestamp and memory type when fields are absent."""
        document = Document(id="memory-2", text="Fallback", metadata={"topic": "unit"})
        restored = MemoryRecord.from_document(document)

        assert restored.memory_type == "long_term"
        assert restored.timestamp.tzinfo is UTC
        assert restored.tags == []
        assert restored.metadata == {"topic": "unit"}


class TestVectorStoreBase:
    """Tests for default VectorStore behavior."""

    def test_add_memories_delegates_documents_and_embeddings(self) -> None:
        """Store memory records and forward explicit embeddings."""
        store = ConcreteVectorStore()
        embeddings = [[0.1, 0.2]]

        ids = store.add_memories([MemoryRecord(id="memory-1", content="Test")], embeddings=embeddings)

        assert ids == ["memory-1"]
        assert store.embeddings == embeddings
        assert store.documents[0].metadata["memory_type"] == "long_term"

    def test_get_by_ids_is_not_implemented(self) -> None:
        """Raise when direct ID retrieval is not supported by a store."""
        store = ConcreteVectorStore()

        with pytest.raises(NotImplementedError, match="Direct ID retrieval"):
            store.get_by_ids(["memory-1"])
