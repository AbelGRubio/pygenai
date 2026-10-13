"""Fixed-size chunking strategy.

========================================================================================================================
Name:         pygenai/rag/chunking/fixed_size.py
Description:  Splits text into fixed-size windows with overlap.
Project:      Pygenai
Date:         2026-10-02 12:29:54
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from pygenai.core.data import Chunk, Document
from pygenai.rag.chunking.base import ChunkStrategy


@dataclass(slots=True)
class FixedSizeChunker(ChunkStrategy):
    """Partition text into chunks of fixed character size with overlap."""

    chunk_size: int = 800
    overlap: int = 80

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document by character count while preserving overlap."""
        if self.chunk_size <= 0:
            msg = "chunk_size must be greater than zero."
            raise ValueError(msg)
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            msg = "overlap must be between 0 and chunk_size - 1."
            raise ValueError(msg)

        text = document.text
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            fragment = text[start:end]
            chunks.append(
                Chunk(
                    id=f"{document.id}-chunk-{index}",
                    text=fragment.strip(),
                    metadata={**document.metadata, "source": document.source},
                    document_id=document.id,
                    chunk_index=index,
                    start=start,
                    end=end,
                )
            )
            index += 1
            start += self.chunk_size - self.overlap
        return [chunk for chunk in chunks if chunk.text]
