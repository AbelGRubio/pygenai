"""Markdown document loader.

========================================================================================================================
Name:         pygenai/rag/loaders/markdown.py
Description:  Parses Markdown files preserving source metadata.
Project:      Pygenai
Date:         2026-10-02 12:29:54
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from pathlib import Path

from pygenai.core.data import Document
from pygenai.rag.loaders.base import FileDocumentLoader


class MarkdownLoader(FileDocumentLoader):
    """Loader for Markdown files that preserves document structure."""

    file_extensions = (".md", ".markdown")

    def load(self, source: str | Path, *, encoding: str = "utf-8") -> list[Document]:
        """Read a markdown file and keep the original heading structure."""
        path = Path(source)
        raw = self._read_text_file(path, encoding=encoding)
        metadata = self._metadata_from_path(path)
        return [
            Document(
                id=self._build_document_id(path),
                text=raw,
                metadata={**metadata, "format": "markdown"},
                source=str(path),
            )
        ]
