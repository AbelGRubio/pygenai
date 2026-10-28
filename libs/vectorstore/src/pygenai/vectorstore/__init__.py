"""Vector storage abstractions and implementations for RAG and Agent Memory.

========================================================================================================================
Name:         pygenai/vectorstores/__init__.py
Description:  Public exports for in-memory and external vector store integrations.
Project:      Pygenai
Date:         2026-10-22 17:16:00
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from .base import MemoryRecord, VectorStore
from .chroma import ChromaVectorStore
from .faiss import FAISSVectorStore
from .in_memory import InMemoryVectorStore
from .memory import AgentMemoryManager
from .pinecone import PineconeVectorStore
from .qdrant import QdrantVectorStore

__all__ = [
    "AgentMemoryManager",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "InMemoryVectorStore",
    "MemoryRecord",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "VectorStore",
]
