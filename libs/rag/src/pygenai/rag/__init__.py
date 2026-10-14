"""Public export surface for the Pygenai RAG package.

========================================================================================================================
Name:         pygenai/rag/__init__.py
Description:  Re-export the package's core types and section-specific modules.
Project:      Pygenai
Date:         2026-10-02 12:19:24
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from .chunking import (
    ChunkStrategy,
    FixedSizeChunker,
    SemanticChunker,
    StructureChunker,
    TokenChunker,
)
from .embeddings import BaseEmbeddings, LiteLLMEmbeddings, SimpleEmbeddings
from .loaders import (
    CSVLoader,
    DocumentLoader,
    FileDocumentLoader,
    MarkdownLoader,
    MultiFormatLoader,
    PDFLoader,
    TextLoader,
    WebLoader,
)
from .retrievers import (
    CrossEncoderReranker,
    HybridRetriever,
    NoOpReranker,
    VectorRetriever,
)
from .vectorstores import (
    ChromaVectorStore,
    FAISSVectorStore,
    InMemoryVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    VectorStore,
)

__version__ = "0.1.0"

__all__ = [
    "BaseEmbeddings",
    "CSVLoader",
    "ChromaVectorStore",
    "ChunkStrategy",
    "CrossEncoderReranker",
    "DocumentLoader",
    "FAISSVectorStore",
    "FileDocumentLoader",
    "FixedSizeChunker",
    "HybridRetriever",
    "InMemoryVectorStore",
    "LiteLLMEmbeddings",
    "MarkdownLoader",
    "MultiFormatLoader",
    "NoOpReranker",
    "PDFLoader",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "SemanticChunker",
    "SimpleEmbeddings",
    "StructureChunker",
    "TextLoader",
    "TokenChunker",
    "VectorRetriever",
    "VectorStore",
    "WebLoader",
]
