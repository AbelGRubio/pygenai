"""Main entry point for the CopilotKit agent graph."""

from pathlib import Path

from langchain_mcp_adapters.sessions import StreamableHttpConnection

from pygenai.agent.agent_builder import AgentBuilder
from pygenai.agent.settings import get_settings
from pygenai.core.logger import get_logger
from pygenai.memory.buffer_memory import BufferMemory
from pygenai.rag.chunking import FixedSizeChunker
from pygenai.rag.loaders import TextLoader
from pygenai.rag.retrievers import HybridRetriever
from pygenai.rag.vectorstores import InMemoryVectorStore

logger = get_logger(__name__)
settings = get_settings()

agent_builder = AgentBuilder()
memory = BufferMemory()
source = Path("./example_documents/rag_demo.txt")

loaded_documents = TextLoader().load(source)
chunker = FixedSizeChunker(chunk_size=220, overlap=40)

chunks = []
for document in loaded_documents:
    chunks.extend(chunker.chunk_document(document))

store = InMemoryVectorStore()
store.add_documents(chunks)

retriever = HybridRetriever(store)

agent_builder.add_rag_source("demo_source", retriever=retriever)
agent_builder.add_memory(memory)

agent_builder.add_mcp(
    name="demo_mcp", mcp=StreamableHttpConnection(url="http://localhost:8000/mcp", transport="streamable_http")
)


# Compile the workflow graph
graph = agent_builder.build()
