"""Example: loading raw files and converting them into RAG documents."""

from __future__ import annotations

from pathlib import Path

from pygenai.rag.loaders import MultiFormatLoader, TextLoader


def main() -> None:
    """Create a sample text file and load it through the supported loaders."""
    path = Path("./example_documents/sample.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Pygenai makes document pipelines easy.\nIt supports loaders, chunkers, and retrievers.", encoding="utf-8"
    )

    text_documents = TextLoader().load(path)
    multi_documents = MultiFormatLoader().load(path)

    print("TextLoader:", text_documents[0].text)
    print("MultiFormatLoader:", multi_documents[0].text)


if __name__ == "__main__":
    main()
