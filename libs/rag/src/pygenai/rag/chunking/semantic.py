"""Semantic chunking strategy.

========================================================================================================================
Name:         pygenai/rag/chunking/semantic.py
Description:  Builds semantically coherent chunks using sentence boundaries.
Project:      Pygenai
Date:         2026-10-02 12:29:54
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pygenai.core.data import Chunk, Document
from pygenai.rag.chunking.base import ChunkStrategy


@dataclass(slots=True)
class SemanticChunker(ChunkStrategy):
    """Split text by sentence boundaries and coalesce into semantically coherent blocks."""

    max_chars: int = 1200
    sentence_pattern: str = r"(?<=[.!?])\s+"

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Build chunks based on sentence windows and a max character budget."""
        sentences = [segment.strip() for segment in re.split(self.sentence_pattern, document.text) if segment.strip()]
        chunks: list[Chunk] = []
        current: list[str] = []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(sentence)
            if current and current_length + sentence_length > self.max_chars:
                joined = " ".join(current)
                chunks.append(
                    Chunk(
                        id=f"{document.id}-chunk-{len(chunks)}",
                        text=joined,
                        metadata={**document.metadata, "source": document.source},
                        document_id=document.id,
                        chunk_index=len(chunks),
                    )
                )
                current = [sentence]
                current_length = sentence_length
            else:
                current.append(sentence)
                current_length += sentence_length

        if current:
            chunks.append(
                Chunk(
                    id=f"{document.id}-chunk-{len(chunks)}",
                    text=" ".join(current),
                    metadata={**document.metadata, "source": document.source},
                    document_id=document.id,
                    chunk_index=len(chunks),
                )
            )

        return chunks
