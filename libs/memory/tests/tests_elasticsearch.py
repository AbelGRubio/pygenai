"""Unit tests for the ElasticsearchMemory semantic search adapter."""

from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest

from pygenai.core.models import Message, MessageRole
from pygenai.memory.elasticsearch_memory import ElasticsearchMemory


class TestElasticsearchMemory:
    """Test suite for ElasticsearchMemory adapter."""

    @pytest.fixture
    def mock_elasticsearch(self) -> MagicMock:
        """Fixture to mock the Elasticsearch client."""
        with patch("pygenai.memory.elasticsearch_memory.Elasticsearch") as mock_es_class:
            mock_client = MagicMock()
            mock_es_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def memory_adapter(self, mock_elasticsearch: MagicMock) -> ElasticsearchMemory:
        """Fixture providing an initialized ElasticsearchMemory instance."""
        return ElasticsearchMemory(
            host="localhost",
            port=9200,
            index_prefix="test-memory",
            embedding_dim=1536,
        )

    @pytest.fixture
    def sample_message(self) -> Message:
        """Fixture providing a sample Message instance."""
        return Message(
            id="msg-123",
            type=MessageRole.USER,
            content="Hello, how can I use Elasticsearch?",
            tokens=10,
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            metadata={"source": "cli"},
        )

    def test_init_defaults(self) -> None:
        """Test initialization with default parameters."""
        with patch("pygenai.memory.elasticsearch_memory.Elasticsearch") as mock_es_class:
            ElasticsearchMemory()
            mock_es_class.assert_called_once_with([{"host": "localhost", "port": 9200}])

    def test_get_index_name(self, memory_adapter: ElasticsearchMemory) -> None:
        """Test index name generation formatting."""
        index_name = memory_adapter._get_index_name("Session-ABC")
        assert index_name == "test-memory-session-abc"

    def test_ensure_index_already_exists(
        self, memory_adapter: ElasticsearchMemory, mock_elasticsearch: MagicMock
    ) -> None:
        """Test _ensure_index when the index already exists."""
        mock_elasticsearch.indices.exists.return_value = True

        memory_adapter._ensure_index("session-1")

        mock_elasticsearch.indices.exists.assert_called_once_with(
            index="test-memory-session-1"
        )
        mock_elasticsearch.indices.create.assert_not_called()

    def test_ensure_index_creates_new(
        self, memory_adapter: ElasticsearchMemory, mock_elasticsearch: MagicMock
    ) -> None:
        """Test _ensure_index when the index does not exist yet."""
        mock_elasticsearch.indices.exists.return_value = False

        memory_adapter._ensure_index("session-1")

        mock_elasticsearch.indices.exists.assert_called_once_with(
            index="test-memory-session-1"
        )
        mock_elasticsearch.indices.create.assert_called_once()
        _, kwargs = mock_elasticsearch.indices.create.call_args
        assert kwargs["index"] == "test-memory-session-1"
        assert "mappings" in kwargs["body"]

    @pytest.mark.asyncio
    async def test_index_message(
        self,
        memory_adapter: ElasticsearchMemory,
        mock_elasticsearch: MagicMock,
        sample_message: Message,
    ) -> None:
        """Test indexing a message successfully."""
        mock_elasticsearch.indices.exists.return_value = True

        await memory_adapter.index_message("session-1", sample_message)

        mock_elasticsearch.index.assert_called_once()
        _, kwargs = mock_elasticsearch.index.call_args
        assert kwargs["index"] == "test-memory-session-1"
        assert kwargs["id"] == "msg-123"
        assert kwargs["body"]["content"] == sample_message.content
        assert kwargs["body"]["type"] == sample_message.type

    @pytest.mark.asyncio
    async def test_search_index_not_exists(
        self, memory_adapter: ElasticsearchMemory, mock_elasticsearch: MagicMock
    ) -> None:
        """Test search returns empty list if index does not exist."""
        mock_elasticsearch.indices.exists.return_value = False

        results = await memory_adapter.search("session-1", "query text")

        assert results == []
        mock_elasticsearch.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_success(
        self, memory_adapter: ElasticsearchMemory, mock_elasticsearch: MagicMock
    ) -> None:
        """Test search successfully retrieves and parses messages."""
        mock_elasticsearch.indices.exists.return_value = True
        mock_elasticsearch.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "message_id": "msg-1",
                            "type": "user",
                            "content": "Elasticsearch vector search",
                            "tokens": 5,
                            "metadata": {"custom": "data"},
                        }
                    }
                ]
            }
        }

        results = await memory_adapter.search("session-1", "vector search", top_k=2)

        assert len(results) == 1
        assert results[0].id == "msg-1"
        assert results[0].content == "Elasticsearch vector search"
        assert results[0].tokens == 5
        assert results[0].metadata == {"custom": "data"}

        mock_elasticsearch.search.assert_called_once()
        _, kwargs = mock_elasticsearch.search.call_args
        assert kwargs["index"] == "test-memory-session-1"
        assert kwargs["body"]["size"] == 2

    @pytest.mark.asyncio
    async def test_clear_index_exists(
        self, memory_adapter: ElasticsearchMemory, mock_elasticsearch: MagicMock
    ) -> None:
        """Test clearing index when it exists."""
        mock_elasticsearch.indices.exists.return_value = True

        await memory_adapter.clear_index("session-1")

        mock_elasticsearch.indices.delete.assert_called_once_with(
            index="test-memory-session-1"
        )

    @pytest.mark.asyncio
    async def test_clear_index_not_exists(
        self, memory_adapter: ElasticsearchMemory, mock_elasticsearch: MagicMock
    ) -> None:
        """Test clearing index when it does not exist."""
        mock_elasticsearch.indices.exists.return_value = False

        await memory_adapter.clear_index("session-1")

        mock_elasticsearch.indices.delete.assert_not_called()

    def test_close(
        self, memory_adapter: ElasticsearchMemory, mock_elasticsearch: MagicMock
    ) -> None:
        """Test closing the Elasticsearch client connection."""
        memory_adapter.close()
        mock_elasticsearch.close.assert_called_once()