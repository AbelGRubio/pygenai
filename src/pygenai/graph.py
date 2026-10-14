"""Main entry point for the CopilotKit agent graph."""

from pathlib import Path
from typing import Any

from pygenai.agent.agent_definition import BaseAgentDefinition
from pygenai.memory.buffer_memory import BufferMemory
from pygenai.rag.chunking import FixedSizeChunker
from pygenai.rag.loaders import TextLoader
from pygenai.rag.retrievers import HybridRetriever
from pygenai.rag.vectorstores import InMemoryVectorStore


class DemoCopilotAgent(BaseAgentDefinition):
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


# Starting point to create the agent graph based on the DemoCopilotAgent definition.
agent_definition = DemoCopilotAgent()
graph = agent_definition.create_graph()