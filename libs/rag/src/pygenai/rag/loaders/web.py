"""Web loader for HTML pages and remote documents.

========================================================================================================================
Name:         pygenai/rag/loaders/web.py
Description:  Downloads and cleans HTML pages so they can be turned into Document objects.
Project:      Pygenai
Date:         2026-10-02 12:19:24
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib import request

from pygenai.core.data import Document
from pygenai.rag.loaders.base import DocumentLoader


class _HTMLTextExtractor(HTMLParser):
    """Small HTML parser used to extract visible text from a page."""

    def __init__(self) -> None:
        """Initialize the parser and a buffer for extracted text."""
        super().__init__()
        self._sections: list[str] = []

    def handle_data(self, data: str) -> None:
        """Add non-markup text to the extraction buffer."""
        text = " ".join(data.split())
        if text:
            self._sections.append(text)

    def get_text(self) -> str:
        """Return the cleaned visible text content."""
        return "\n".join(self._sections)


class WebLoader(DocumentLoader):
    """Download a remote HTML document and convert it to a Document."""

    def load(self, source: str | Path, *, encoding: str = "utf-8") -> list[Document]:
        """Fetch a web page and create a single document from its content."""
        url = str(source)
        with request.urlopen(request.Request(url, headers={"User-Agent": "pygenai-rag/1.0"})) as response:  # noqa: S310
            raw_html = response.read().decode(encoding, errors="replace")

        parser = _HTMLTextExtractor()
        parser.feed(raw_html)
        text = parser.get_text()
        metadata = {"url": url, "source_type": "web"}
        return [Document(id=f"web-{url}", text=text, metadata=metadata, source=url)]


__all__ = ["WebLoader"]
