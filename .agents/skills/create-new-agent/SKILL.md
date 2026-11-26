---
name: create-new-agent
description: 'Create a new AI agent with the Pygenai tech stack, code conventions, and Makefile patterns. Use when writing Python code, adding CLI commands, defining MCP tools, editing Makefiles, adding dependencies, or reviewing code in this repo. Covers: fastmcp, typer, pydantic, uv, ruff, pyrefly, pytest, Python 3.13, type annotations, imports, docstrings, logging patterns.'
---
# Skill: PyGenAI Agent Creator (`pygenai-agent-creator`)

## Overview
This skill provides guidelines, best practices, and code templates for building modular AI agents using the `pygenai` ecosystem. The ecosystem consists of specialized packages designed to handle distinct parts of an agent's lifecycle, memory, context management, and reasoning logic.

---

## Ecosystem Architecture

The `pygenai` suite relies on the following core libraries:

* **`pygenai-core`**: Provides shared domain models, base schemas (`BaseMessage`, `ModelConfig`), logging, utilities, and exception handling.
* **`pygenai-llm`**: Manages unified LLM provider abstractions (OpenAI, Anthropic, Azure, AWS Bedrock, Ollama) with intelligent fallback routing, retry strategies, and token cost tracking.
* **`pygenai-agent`**: Provides core node execution engines (`AgentNodeExecutor`), pipeline builders (`AgentBuilder`), and base agent abstractions (`BaseAgentDefinition`).
* **`pygenai-memory`**: Manages short-term and conversational state through strategies like `BufferMemory`, `WindowMemory`, and `SummarizedMemory` backed by adapters (`InMemoryAdapter`, `RedisAdapter`, `PostgreSQLAdapter`).
* **`pygenai-vectorstore`**: Provides vector collection managers and multi-tier long-term semantic storage (`AgentMemoryManager`, `ChromaVectorStore`, `FAISSVectorStore`, `QdrantVectorStore`, `InMemoryVectorStore`).
* **`pygenai-rag`**: Handles end-to-end Retrieval-Augmented Generation workflows, including document loaders (`TextLoader`, `PDFLoader`), chunkers (`FixedSizeChunker`), embedders, and retrievers (`VectorRetriever`, `HybridRetriever`).

---

## Agent Creation Guidelines

When constructing an agent with `pygenai`:

1. **Inherit from `BaseAgentDefinition`**: Define your custom agent class by inheriting from `BaseAgentDefinition`.
2. **Configure Memory (`configure_memory`)**: Instantiate and return the required conversational memory structure (e.g., `BufferMemory` or `WindowMemory`).
3. **Configure RAG Sources (`configure_rag`)**: Load documents, chunk text, index into a vector store backend, and return named retriever mappings.
4. **Configure MCP Connections (`configure_mcps`)**: Optionally declare Model Context Protocol endpoints (e.g., `StreamableHttpConnection`).
5. **Construct Execution Logic**: Use either the default `create_graph()` method or override `AgentBuilder` to define a custom node graph (e.g., custom reasoning steps or memory persist nodes).

---

## Code Examples

### Example 1: Standard Agent with RAG, Memory, and MCP (`graph.py`)

This example demonstrates setting up a standard graph agent using `BaseAgentDefinition` equipped with an in-memory buffer, hybrid RAG retrieval over plain text documents, and an external MCP connection.

```python
"""Standard Copilot Agent Definition using pygenai."""

from pathlib import Path
from typing import Any

from langchain_mcp_adapters.sessions import StreamableHttpConnection

from pygenai.agent.base_definition import BaseAgentDefinition
from pygenai.memory.buffer_memory import BufferMemory
from pygenai.rag.chunking import FixedSizeChunker
from pygenai.rag.loaders import TextLoader
from pygenai.rag.retrievers import HybridRetriever
from pygenai.vectorstore import InMemoryVectorStore


class DemoCopilotAgent(BaseAgentDefinition):
    """Custom agent definition integrating RAG, Memory, and MCP connections."""

    def configure_memory(self) -> BufferMemory:
        """Configure short-term conversational buffer memory."""
        return BufferMemory()

    def configure_rag(self) -> dict[str, Any]:
        """Configure RAG retriever pipelines over local document stores."""
        source = Path("./example_documents/rag_demo.txt")
        loaded_documents = TextLoader().load(source)
        chunker = FixedSizeChunker(chunk_size=220, overlap=40)

        chunks = []
        for document in loaded_documents:
            chunks.extend(chunker.chunk_document(document))

        store = InMemoryVectorStore()
        store.add_documents(chunks)
        retriever = HybridRetriever(store)

        return {"demo_source": retriever}

    def configure_mcps(self) -> dict[str, Any]:
        """Configure Model Context Protocol (MCP) server endpoints."""
        return {
            "demo_mcp": StreamableHttpConnection(
                url="http://localhost:8000/mcp",
                transport="streamable_http"
            )
        }


# Instantiate definition and create runnable execution graph
agent_definition = DemoCopilotAgent()
graph = agent_definition.create_graph()
```

---

### Example 2: Custom Workflow Agent (`custom_workflow.py`)

This example illustrates how to build an agent with non-standard execution paths (e.g., custom reasoning, explicit pre/post-memory operations) by extending `AgentBuilder` and utilizing `AgentNodeExecutor`.

```python
"""Custom Workflow Agent Definition using pygenai."""

from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pygenai.agent.base_definition import BaseAgentDefinition
from pygenai.agent.builder import AgentBuilder
from pygenai.agent.node_executor import AgentNodeExecutor
from pygenai.memory import BufferMemory
from pygenai.rag import FixedSizeChunker, HybridRetriever, TextLoader
from pygenai.vectorstore import InMemoryVectorStore


class CustomBuilder(AgentBuilder):
    """Custom AgentBuilder overriding graph compilation for tailored execution steps."""

    def __init__(self, state_schema: Any = None) -> None:
        super().__init__(state_schema=state_schema)

    def build(self) -> CompiledStateGraph:
        """Build and compile custom StateGraph workflow nodes."""
        executor = AgentNodeExecutor(
            llm=self._llm,
            rag_sources=self._rag_sources,
            memory=self._memory,
            skills=self._skills,
            mcps=self._mcps,
        )

        workflow = StateGraph(self._state_schema)

        # Register custom execution nodes
        workflow.add_node("prepare_rag_query", executor.prepare_rag_query_node)
        workflow.add_node("memory_node", executor.memory_node)
        workflow.add_node("reasoning", executor.reasoning_node)
        workflow.add_node("save_memory", executor.save_memory_node)

        # Define custom node execution order
        workflow.set_entry_point("prepare_rag_query")
        workflow.add_edge("prepare_rag_query", "memory_node")
        workflow.add_edge("memory_node", "reasoning")
        workflow.add_edge("reasoning", "save_memory")
        workflow.add_edge("save_memory", END)

        return workflow.compile()


class CustomWorkflowAgent(BaseAgentDefinition):
    """Agent definition that attaches the custom workflow logic."""

    def configure_memory(self) -> BufferMemory:
        """Configure memory buffer."""
        return BufferMemory()

    def configure_rag(self) -> dict[str, Any]:
        """Configure RAG retriever pipelines."""
        source = Path("./example_documents/rag_demo.txt")
        loaded_documents = TextLoader().load(source)
        chunker = FixedSizeChunker(chunk_size=220, overlap=40)

        chunks = []
        for document in loaded_documents:
            chunks.extend(chunker.chunk_document(document))

        store = InMemoryVectorStore()
        store.add_documents(chunks)
        retriever = HybridRetriever(store)

        return {"demo_source": retriever}


# Create instance and graph
agent_definition = CustomWorkflowAgent()
graph = agent_definition.create_graph()
```
