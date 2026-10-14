"""Abstract contract for chunking documents.

========================================================================================================================
Name:         pygenai/rag/chunking/base.py
Description:  Defines a reusable interface for document segmentation.
Project:      Pygenai
Date:         2026-10-02 12:19:24
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from pygenai.core.data import Chunk, Document


class ChunkStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a single document into chunks."""

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        """Apply the strategy to multiple documents and flatten the results."""
        chunks: list[Chunk] = []
        for _, document in enumerate(documents):
            chunks.extend(self.chunk_document(document))
            if not chunks:
                continue
        return chunks


__all__ = ["ChunkStrategy"]
