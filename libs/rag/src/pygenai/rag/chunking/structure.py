"""Structure-aware chunking strategy.

========================================================================================================================
Name:         pygenai/rag/chunking/structure.py
Description:  Segments text by document structure, such as headings and sections.
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
class StructureChunker(ChunkStrategy):
    """Split Markdown-like documents by heading sections and paragraphs."""

    max_chars: int = 1500

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Create chunks based on section heading boundaries."""
        lines = document.text.splitlines()
        section: list[str] = []
        current_title: str | None = None
        chunks: list[Chunk] = []

        def flush() -> None:
            if not section:
                return
            text = "\n".join(section).strip()
            if not text:
                return
            chunks.append(
                Chunk(
                    id=f"{document.id}-chunk-{len(chunks)}",
                    text=text,
                    metadata={**document.metadata, "source": document.source, "section": current_title},
                    document_id=document.id,
                    chunk_index=len(chunks),
                )
            )

        for line in lines:
            if re.match(r"^#{1,6}\s+.+", line):
                if section:
                    flush()
                    section = []
                current_title = line.lstrip("#").strip()
                section.append(line)
            else:
                section.append(line)

            if sum(len(part) for part in section) > self.max_chars and section:
                flush()
                section = []

        flush()
        return chunks
