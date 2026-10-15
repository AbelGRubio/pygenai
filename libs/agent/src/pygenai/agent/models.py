from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


@dataclass
class LLMConfig:
    """Configuration class for LLM instances."""
    provider: str
    model_name: str
    api_base: str | None = None
    temperature: float = 0.7
    timeout: float | None = None
    retry_config: dict[str, Any] = field(default_factory=lambda: {"max_retries": 2})


class AgentState(TypedDict):
    """Defines the expanded state structure for the LangGraph agent."""
    messages: list[BaseMessage]
    context: dict[str, Any]
    rag_query: str | None
    rag_texts: list[str]
    tools: list[Any]
    memory_data: dict[str, Any]
    session_id: str | None
