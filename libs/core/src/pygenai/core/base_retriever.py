"""Base retriever interfaces and result model.

========================================================================================================================
Name:         pygenai/rag/retrievers/base.py
Description:  Defines the core retrieval contract and a typed result object.
Project:      Pygenai
Date:         2026-10-02 12:19:24
Status:       Development

Copyright ©2026 All rights reserved.
========================================================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pygenai.core.data import Chunk, Document


@dataclass(slots=True)
class RetrievedDocument:
    """A retrieval result with a combined source document and score."""

    document: Document | Chunk
    score: float
    rank: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract interface for retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve the top-k results for a query."""

    def retrieve_and_format(self, query: str, top_k: int = 5) -> str:
        """Retrieve documents and format them for downstream consumption."""
        raise NotImplementedError


__all__ = ["BaseRetriever", "RetrievedDocument"]
