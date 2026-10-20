"""Agent Builder. """
from __future__ import annotations

from typing import Any, TypeAlias

from langchain_litellm import ChatLiteLLM
from langchain_mcp_adapters.sessions import (
    SSEConnection,
    StdioConnection,
    StreamableHttpConnection,
    WebsocketConnection,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pygenai.agent.node_executor import AgentNodeExecutor
from pygenai.agent.models import AgentState, LLMConfig
from pygenai.agent.settings import get_settings
from pygenai.core.base_memory import MemoryTypes
from pygenai.core.base_retriever import BaseRetriever
from pygenai.core.logger import get_logger

logger = get_logger(__name__)
MCPConnection: TypeAlias = StdioConnection | SSEConnection | StreamableHttpConnection | WebsocketConnection


class AgentBuilder:
    """Builder class focused purely on fluent configuration and graph assembly."""

    def __init__(self, state_schema: Any = AgentState) -> None:
        """Default constructor for the AgentBuilder. Initializes the builder with default settings."""
        self._state_schema = state_schema
        self._default_model: str = "openai/gpt-4o-mini"
        self._rag_sources: dict[str, BaseRetriever] = {}
        self._memory: MemoryTypes | None = None
        self._skills: list[Any] = []
        self._mcps: dict[str, MCPConnection] = {}
        self._llm: ChatLiteLLM
        self._define_default_llm()

    def _define_default_llm(self) -> None:
        """Define the default LLM configuration using environment settings."""
        settings = get_settings()
        self._llm = ChatLiteLLM(
            model=self._default_model,
            temperature=0.7,
            api_key=settings.llm_api_key.get_secret_value(),
            api_base=settings.model_base_url,
        )

    def with_llm(self, config: LLMConfig) -> AgentBuilder:
        """Configure the agent with a specific LLM configuration."""
        self._llm = ChatLiteLLM(
            model=config.model_name,
            temperature=config.temperature,
            api_base=config.api_base,
            api_key=get_settings().llm_api_key.get_secret_value(),
        )
        return self

    def add_rag_source(self, name: str, retriever: BaseRetriever) -> AgentBuilder:
        """Add a RAG source to the agent. The name is used to identify the source in the workflow."""
        self._rag_sources[name] = retriever
        return self

    def add_memory(self, memory: MemoryTypes) -> AgentBuilder:
        """Add a memory system to the agent."""
        self._memory = memory
        return self

    def add_skill(self, skill: Any) -> AgentBuilder:
        """Add a skill to the agent. Skills can be any callable or object that the agent can use."""
        self._skills.append(skill)
        return self

    def add_mcp(self, name: str, mcp: MCPConnection) -> AgentBuilder:
        """Add a Multi-Channel Protocol (MCP) connection to the agent.
        The name is used to identify the MCP in the workflow."""
        self._mcps[name] = mcp
        return self

    def build(self) -> CompiledStateGraph:
        """Build and compile the LangGraph workflow.
        
        Subclasses can inherit this builder and override this method to change 
        how nodes are wired or execute custom reasoning flows.
        """
        executor = AgentNodeExecutor(
            llm=self._llm,
            rag_sources=self._rag_sources,
            memory=self._memory,
            skills=self._skills,
            mcps=self._mcps
        )

        workflow = StateGraph(self._state_schema)

        # Register nodes using the executor methods
        workflow.add_node("prepare_rag_query", executor.prepare_rag_query_node)
        workflow.add_node("prepare_tools", executor.prepare_tools_node)
        workflow.add_node("memory_node", executor.memory_node)
        workflow.add_node("reasoning", executor.reasoning_node)
        workflow.add_node("save_memory", executor.save_memory_node)

        # Default linear execution flow
        workflow.set_entry_point("prepare_rag_query")
        workflow.add_edge("prepare_rag_query", "memory_node")
        workflow.add_edge("memory_node", "reasoning")
        workflow.add_edge("reasoning", "save_memory")
        workflow.add_edge("save_memory", END)

        return workflow.compile()
