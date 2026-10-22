---
name: pygenai-rag-example-builder
description: Generates RAG pipeline examples and code using the pygenai library, supporting pre-loaded/persistent vector stores (InMemory, Chroma, FAISS, Pinecone, Qdrant) with explicit query embedding computation via LiteLLMEmbeddings or SimpleEmbeddings. Use when the user asks to create a RAG example or pipeline with pygenai.
---
# Pygenai RAG Example Builder

A skill to guide the creation of Retrieval-Augmented Generation (RAG) pipelines and examples using the `pygenai` package. It covers document loading, chunking, embedding generation using `LiteLLMEmbeddings` or `SimpleEmbeddings`, connecting to pre-loaded or persistent vector stores, and performing similarity searches.

## Embedding Providers

When querying vector stores that operate directly on numerical representations (such as Pinecone, Qdrant, or FAISS), convert text queries into vector embeddings prior to search using one of the available embedding providers in `pygenai.rag.embeddings`:

1. **`LiteLLMEmbeddings`**: Uses LLM embedding models (e.g., Gemini, OpenAI).
2. **`SimpleEmbeddings`**: A deterministic, offline bag-of-words embedder suitable for testing/development.

### Query Embedding Example Pattern

```python
from pygenai.rag.embeddings import LiteLLMEmbeddings, SimpleEmbeddings

# Option 1: LiteLLM Embeddings (Remote/API-backed)
embedder = LiteLLMEmbeddings(model="gemini/gemini-embedding-2")

# Option 2: Simple Embeddings (Offline / Local testing)
# embedder = SimpleEmbeddings()

# Calculate query vector embedding
query_text = "What types of loaders are supported?"
query_vector = embedder.embed_query(query_text)
```

---

## Vector Stores Architecture & Similarity Search Examples

### 1. PineconeVectorStore (Cloud Index + Query Embedding)

```python
from pygenai.rag.embeddings import LiteLLMEmbeddings, SimpleEmbeddings
from pygenai.rag.vectorstores import PineconeVectorStore

# 1. Initialize embedder
embedder = LiteLLMEmbeddings(model="gemini/gemini-embedding-2")
# embedder = SimpleEmbeddings()

# 2. Compute query vector
query = "What is machine learning?"
query_vector = embedder.embed_query(query)

# 3. Connect to Pinecone and search using the computed vector
store = PineconeVectorStore(index_name="enterprise-rag", api_key="pinecone-api-key")
results = store.similarity_search(
    query="",
    k=5,
    filter_search={"query_vector": query_vector}
)
```

### 2. QdrantVectorStore (Qdrant Collection + Query Embedding)

```python
from pygenai.rag.embeddings import LiteLLMEmbeddings, SimpleEmbeddings
from pygenai.rag.vectorstores import QdrantVectorStore

# 1. Initialize embedder
embedder = LiteLLMEmbeddings(model="gemini/gemini-embedding-2")
# embedder = SimpleEmbeddings()

# 2. Compute query vector
query = "Explain hybrid search."
query_vector = embedder.embed_query(query)

# 3. Connect to Qdrant and search
store = QdrantVectorStore(collection_name="docs_v1", url="http://localhost:6333")
results = store.similarity_search(
    query="",
    k=3,
    filter_search={"query_vector": query_vector}
)
```

### 3. FAISSVectorStore (Local Index + Query Embedding)

```python
from pygenai.rag.embeddings import LiteLLMEmbeddings, SimpleEmbeddings
from pygenai.rag.vectorstores import FAISSVectorStore

# 1. Initialize embedder
embedder = LiteLLMEmbeddings(model="gemini/gemini-embedding-2")
# embedder = SimpleEmbeddings()

# 2. Compute query vector
query = "How to load CSV files?"
query_vector = embedder.embed_query(query)

# 3. Initialize FAISS store and perform vector search
store = FAISSVectorStore(dimension=len(query_vector), metric="l2")
results = store.similarity_search(
    query="",
    k=3,
    filter_search={"query_embedding": query_vector}
)
```

### 4. ChromaVectorStore (Built-in or Persistent Text Search)

```python
from pygenai.rag.vectorstores import ChromaVectorStore

# Connect to a pre-loaded persistent Chroma database
store = ChromaVectorStore(
    collection_name="knowledge_base",
    persist_directory="./chroma_db"
)

# Search pre-indexed collection directly with query string
results = store.similarity_search(query="What is machine learning?", k=3)
```

### 5. InMemoryVectorStore (Volatile / Local In-Memory)

```python
from pygenai.rag.embeddings import LiteLLMEmbeddings, SimpleEmbeddings
from pygenai.rag.vectorstores import InMemoryVectorStore
from pygenai.core.data import Document

# Uses LiteLLMEmbeddings or defaults to SimpleEmbeddings internally
embedder = LiteLLMEmbeddings(model="gemini/gemini-embedding-2")
# embedder = SimpleEmbeddings()

store = InMemoryVectorStore(embeddings=embedder)
store.add_documents([Document(id="1", text="Pygenai simplifies document RAG pipelines.")])

results = store.similarity_search("RAG pipelines", k=1)
```

---

## End-to-End Startup Application Example

```python
from __future__ import annotations

from pygenai.rag.embeddings import LiteLLMEmbeddings, SimpleEmbeddings
from pygenai.rag.vectorstores import QdrantVectorStore


class QdrantRAGService:
    """RAG Service connecting to a pre-indexed Qdrant vector store."""

    def __init__(self, collection_name: str = "knowledge_base", url: str = "http://localhost:6333"):
        # Select embedding model for queries
        self.embedder = LiteLLMEmbeddings(model="gemini/gemini-embedding-2")
        # self.embedder = SimpleEmbeddings()

        # Connect to pre-loaded vector store on startup
        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            url=url,
        )

    def search(self, question: str, top_k: int = 3) -> list[str]:
        # Step 1: Calculate embedding for user question
        query_vector = self.embedder.embed_query(question)

        # Step 2: Retrieve matching documents from vector store using the vector
        hits = self.vector_store.similarity_search(
            query="",
            k=top_k,
            filter_search={"query_vector": query_vector},
        )
        return [hit.document.text for hit in hits]


if __name__ == "__main__":
    rag_app = QdrantRAGService()
    results = rag_app.search("How do document loaders work?")
    for res in results:
        print("-", res)
```