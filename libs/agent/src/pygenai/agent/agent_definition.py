"""Base class for defining modular agents declaratively."""
from abc import ABC
from typing import Any
from pygenai.agent.agent_builder import AgentBuilder
from langgraph.graph.state import CompiledStateGraph


class BaseAgentDefinition(ABC):
    """Base class for defining modular agents declaratively.

    Subclasses only override the methods corresponding to the features
    they need (RAG, Memory, MCP, LLM). Unimplemented hooks fallback to defaults.
    """

    def configure_llm(self) -> Any | None:
        """Override to specify a custom LLM config, or use default."""
        return None

    def configure_memory(self) -> Any | None:
        """Override to add memory systems."""
        return None

    def configure_rag(self) -> dict[str, Any]:
        """Override to add RAG sources. Returns a dict of {name: retriever}."""
        return {}

    def configure_mcps(self) -> dict[str, Any]:
        """Override to add MCP connections. Returns a dict of {name: connection}."""
        return {}

    def create_graph(self) -> CompiledStateGraph:
        """Assembles the agent using the AgentBuilder based on provided hooks."""
        builder = AgentBuilder()

        # 1. Optional LLM
        llm_config = self.configure_llm()
        if llm_config:
            builder.with_llm(llm_config)

        # 2. Optional Memory
        memory = self.configure_memory()
        if memory:
            builder.add_memory(memory)

        # 3. Optional RAG sources
        for name, retriever in self.configure_rag().items():
            builder.add_rag_source(name, retriever)

        # 4. Optional MCPs
        for name, mcp in self.configure_mcps().items():
            builder.add_mcp(name, mcp)

        return builder.build()
