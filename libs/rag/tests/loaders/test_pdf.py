"""Tests for the PDF document loader."""

from pathlib import Path

import pytest

from pygenai.rag.loaders.pdf import PDFLoader


def test_pdf_loader_extracts_pages(monkeypatch, tmp_path: Path) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return "Page text"

    class FakeReader:
        pages = [FakePage(), FakePage()]

    def fake_pdf_reader(_path: str) -> FakeReader:
        return FakeReader()

    import sys

    fake_pypdf = type("FakePypdf", (), {"PdfReader": fake_pdf_reader})
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)
    source = tmp_path / "guide.pdf"

    documents = PDFLoader().load(source)

    assert len(documents) == 1
    assert documents[0].text == "Page text\n\nPage text"
    assert documents[0].metadata["page_count"] == 2


def test_pdf_loader_reports_missing_dependency(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def missing_pypdf(name: str, *args: object, **kwargs: object) -> object:
        if name == "pypdf":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_pypdf)

    with pytest.raises(ImportError, match="PDF parsing requires"):
        PDFLoader().load("missing.pdf")
