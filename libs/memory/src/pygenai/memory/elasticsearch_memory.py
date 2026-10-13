"""Elasticsearch-based semantic search for conversation memory."""

from elasticsearch import Elasticsearch
from pygenai.core.base_memory import SemanticSearchAdapter
from pygenai.core.models import Message


class ElasticsearchMemory(SemanticSearchAdapter):
    """
    Semantic search adapter using Elasticsearch for hybrid retrieval.

    Enables similarity-based retrieval of past interactions via vector and
    keyword search, allowing agents to recover relevant past context.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_prefix: str = "memory",
        embedding_dim: int = 1536,
    ):
        """
        Initialize ElasticsearchMemory.

        Args:
            host: Elasticsearch host (default: localhost).
            port: Elasticsearch port (default: 9200).
            index_prefix: Prefix for index names (default: memory).
            embedding_dim: Dimension of embeddings (default: 1536 for OpenAI).
        """
        self._client = Elasticsearch([{"host": host, "port": port}])
        self._index_prefix = index_prefix
        self._embedding_dim = embedding_dim

    def _get_index_name(self, session_id: str) -> str:
        """
        Get the Elasticsearch index name for a session.

        Args:
            session_id: The session identifier.

        Returns:
            Index name.
        """
        return f"{self._index_prefix}-{session_id}".lower()

    def _ensure_index(self, session_id: str) -> None:
        """
        Ensure the index exists with proper mappings.

        Args:
            session_id: The session identifier.
        """
        index_name = self._get_index_name(session_id)

        if self._client.indices.exists(index=index_name):
            return

        self._client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "message_id": {"type": "keyword"},
                        "type": {"type": "keyword"},
                        "content": {"type": "text"},
                        "tokens": {"type": "integer"},
                        "timestamp": {"type": "date"},
                        "metadata": {"type": "object", "enabled": False},
                    }
                }
            },
        )

    async def index_message(self, session_id: str, message: Message) -> None:
        """
        Index a message for semantic search.

        Args:
            session_id: The session identifier.
            message: The message to index.
        """
        self._ensure_index(session_id)
        index_name = self._get_index_name(session_id)

        doc = {
            "message_id": message.id,
            "type": message.type,
            "content": message.content,
            "tokens": message.tokens,
            "timestamp": message.timestamp.isoformat(),
            "metadata": message.metadata,
        }

        self._client.index(index=index_name, id=message.id, body=doc)

    async def search(self, session_id: str, query: str, top_k: int = 5) -> list[Message]:
        """
        Search for semantically similar messages.

        Args:
            session_id: The session identifier.
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            List of relevant messages.
        """
        index_name = self._get_index_name(session_id)

        if not self._client.indices.exists(index=index_name):
            return []

        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "metadata"],
                }
            },
            "size": top_k,
            "sort": [{"timestamp": {"order": "desc"}}],
        }

        results = self._client.search(index=index_name, body=search_body)
        messages = []

        for hit in results["hits"]["hits"]:
            source = hit["_source"]
            message = Message(
                id=source["message_id"],
                type=source["type"],
                content=source["content"],
                tokens=source.get("tokens", 0),
                metadata=source.get("metadata", {}),
            )
            messages.append(message)

        return messages

    async def clear_index(self, session_id: str) -> None:
        """
        Clear the search index for a session.

        Args:
            session_id: The session identifier.
        """
        index_name = self._get_index_name(session_id)
        if self._client.indices.exists(index=index_name):
            self._client.indices.delete(index=index_name)

    def close(self) -> None:
        """Close the Elasticsearch connection."""
        self._client.close()
