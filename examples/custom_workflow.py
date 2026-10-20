"""Custom workflow agent definition using RAG and Memory (MCP is omitted)."""

from pathlib import Path
from typing import Any

from agent.node_executor import AgentNodeExecutor
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pygenai.agent.base_definition import BaseAgentDefinition
from pygenai.agent.builder import AgentBuilder
from pygenai.memory import BufferMemory
from pygenai.rag import FixedSizeChunker, HybridRetriever, InMemoryVectorStore, TextLoader


class CustomBuilder(AgentBuilder):
    """Custom AgentBuilder that can be extended for additional configurations."""

    def __init__(self, state_schema: Any = None) -> None:
        """Initialize the CustomBuilder with an optional state schema."""
        super().__init__(state_schema=state_schema)

    def build(self) -> CompiledStateGraph:
        """Builds the agent graph based on the current configuration."""
        executor = AgentNodeExecutor(
            llm=self._llm, rag_sources=self._rag_sources, memory=self._memory, skills=self._skills, mcps=self._mcps
        )

        workflow = StateGraph(self._state_schema)

        # Register nodes using the executor methods
        workflow.add_node("prepare_rag_query", executor.prepare_rag_query_node)
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


class CustomWorkflowAgent(BaseAgentDefinition):
    """Custom agent definition using RAG and Memory (MCP is omitted)."""

    def configure_memory(self) -> BufferMemory:
        """Configure a simple in-memory buffer for the agent."""
        return BufferMemory()

    def configure_rag(self) -> dict[str, Any]:
        """Configure a RAG source for the agent using a text file."""
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


agent_definition = CustomWorkflowAgent()
graph = agent_definition.create_graph()
