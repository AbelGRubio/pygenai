"""Token-aware chunking strategy.

========================================================================================================================
Name:         pygenai/rag/chunking/token.py
Description:  Splits text based on token-like units approximated by whitespace.
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
class TokenChunker(ChunkStrategy):
    """Partition text by token-like units using a word-based approximation."""

    max_tokens: int = 256
    overlap_tokens: int = 32

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text on whitespace as a lightweight proxy for model tokens."""
        return text.split()

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split content into windows sized by token count approximations."""
        tokens = self._tokenize(document.text)
        chunks: list[Chunk] = []
        chunk_size = max(self.max_tokens, 1)
        overlap = max(self.overlap_tokens, 0)
        pointer = 0
        index = 0

        while pointer < len(tokens):
            end = min(pointer + chunk_size, len(tokens))
            fragment = " ".join(tokens[pointer:end])
            chunks.append(
                Chunk(
                    id=f"{document.id}-chunk-{index}",
                    text=fragment,
                    metadata={**document.metadata, "source": document.source},
                    document_id=document.id,
                    chunk_index=index,
                    start=pointer,
                    end=end,
                )
            )
            index += 1
            pointer += chunk_size - overlap

        return [chunk for chunk in chunks if chunk.text]
