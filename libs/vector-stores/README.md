# Pygenai: Vector Stores & Long-Term Memory

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Development](https://img.shields.io/badge/status-development-orange.svg)]()

> A robust, decoupled vector storage and agent memory package designed for modern AI applications, supporting both document retrieval (RAG) and multi-tier agent memory management (short-term/long-term context).

---

## Features

- **Decoupled Architecture:** Clean separation of vector database adapters from specific RAG pipelines, located under the core `pygenai.vectorstores` package.
- **Multi-Tier Agent Memory:** First-class support for short-term session tracking (with TTL/cleanup capabilities) and long-term semantic storage using `MemoryRecord` and `AgentMemoryManager`.
- **Multiple Vector Database Backends:** Unified API abstractions supporting multiple industry-standard engines:
  - **ChromaDB** (`ChromaVectorStore`)
  - **FAISS** (`FAISSVectorStore`)
  - **Pinecone** (`PineconeVectorStore`)
  - **Qdrant** (`QdrantVectorStore`)
  - **In-Memory** (`InMemoryVectorStore` powered by LangChain)
- **Advanced Metadata Filtering:** Native payload filtering for user isolation (`user_id`), conversation tracking (`session_id`), and memory scopes (`memory_type`).
- **Strict Type Hinting & Documentation:** Fully typed Python codebase (`from __future__ import annotations`) documented thoroughly in English.

---

## Architecture and Project Structure

```text
pygenai/
└── vectorstores/
    ├── __init__.py        # Public package exports
    ├── base.py            # Abstract VectorStore interface and MemoryRecord data class
    ├── memory.py          # High-level AgentMemoryManager for short/long term storage
    ├── chroma.py          # ChromaDB integration adapter
    ├── faiss.py           # FAISS local vector store adapter
    ├── in_memory.py       # LangChain-backed in-memory store
    ├── pinecone.py        # Pinecone cloud vector store adapter
    └── qdrant.py          # Qdrant collection and payload filter adapter
```

---

## Prerequisites and Environment

- **Python Version:** Python 3.11 or higher.
- **Package Manager:** `pip` or `uv`.
- **Environment Variables:** Create a `.env` file based on the template below depending on the backend provider you intend to use.

### `.env.example`
```env
# Optional LLM / Embedding Providers
OPENAI_API_KEY=your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here

# Vector Database Credentials (if applicable)
PINECONE_API_KEY=your-pinecone-api-key-here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-api-key-here
CHROMA_PERSIST_DIRECTORY=./chroma_data
```

---

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/pygenai.git
   cd pygenai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies and the package:**
   ```bash
   pip install -e .
   # Or install specific vector store optional dependencies, e.g.:
   pip install chromadb qdrant-client pinecone-client faiss-cpu
   ```

---

## Usage Guide / Quick Examples

### 1. Using Agent Memory (`AgentMemoryManager`)
Initialize a vector store and use the high-level manager to handle conversational short-term memory and persistent long-term knowledge.

```python
from pygenai.vectorstores import InMemoryVectorStore, AgentMemoryManager

# Initialize backend store
store = InMemoryVectorStore()

# Create memory manager facade
memory_mgr = AgentMemoryManager(vector_store=store)

# Save a long-term user preference
memory_mgr.save_memory(
    content="User prefers coding in Python and using async/await patterns.",
    memory_type="long_term",
    user_id="user_123",
    tags=["preferences", "programming"]
)

# Save a short-term session context
memory_mgr.save_memory(
    content="User is currently debugging a vector store indexing issue.",
    memory_type="short_term",
    session_id="sess_abc",
    user_id="user_123"
)

# Recall relevant memories matching a query
results = memory_mgr.recall(
    query="What programming language does the user like?",
    user_id="user_123",
    k=3
)

for hit in results:
    print(f"Score: {hit.score} | Content: {hit.document.text}")
```

### 2. Direct Vector Store Usage (RAG & Documents)
You can also use adapters directly for document ingestion and similarity search in RAG pipelines:

```python
from pygenai.core.data import Document
from pygenai.vectorstores import ChromaVectorStore

# Initialize Chroma vector store with persistence
vector_store = ChromaVectorStore(
    collection_name="knowledge_base",
    persist_directory="./chroma_db"
)

# Ingest documents
documents = [
    Document(id="doc-1", text="ChromaDB is an open-source vector database."),
    Document(id="doc-2", text="Qdrant provides high-performance vector similarity search.")
]

# Note: Pass precomputed embeddings if required by the adapter or let embedding models handle it
vector_store.add_documents(documents)

# Search relevant documents
hits = vector_store.similarity_search(query="Tell me about ChromaDB", k=2)
for hit in hits:
    print(hit.document.text)
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository and create your feature branch (`git checkout -b feature/amazing-feature`).
2. Ensure strict type hints and docstrings are provided in English for all new functions/classes.
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).