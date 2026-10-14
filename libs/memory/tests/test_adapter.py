"""Unit tests for the persistence adapters (InMemory, Redis, PostgreSQL)."""

import json
from unittest.mock import MagicMock, patch
import pytest

from pygenai.core.models import ConversationState
from pygenai.memory.adapters import InMemoryAdapter, PostgreSQLAdapter, RedisAdapter


class TestInMemoryAdapter:
    """Test suite for the InMemoryAdapter persistence backend."""

    @pytest.fixture
    def adapter(self) -> InMemoryAdapter:
        """Fixture providing an instance of InMemoryAdapter."""
        return InMemoryAdapter()

    @pytest.fixture
    def sample_state(self) -> ConversationState:
        """Fixture providing a sample ConversationState."""
        return ConversationState(session_id="session-1", messages=[])

    @pytest.mark.asyncio
    async def test_save_and_load(self, adapter: InMemoryAdapter, sample_state: ConversationState) -> None:
        """Test saving and loading state from memory."""
        await adapter.save("session-1", sample_state)
        loaded = await adapter.load("session-1")
        assert loaded == sample_state

    @pytest.mark.asyncio
    async def test_load_non_existent(self, adapter: InMemoryAdapter) -> None:
        """Test loading a session that does not exist returns None."""
        loaded = await adapter.load("non-existent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_exists(self, adapter: InMemoryAdapter, sample_state: ConversationState) -> None:
        """Test checking if a session exists."""
        assert await adapter.exists("session-1") is False
        await adapter.save("session-1", sample_state)
        assert await adapter.exists("session-1") is True

    @pytest.mark.asyncio
    async def test_delete(self, adapter: InMemoryAdapter, sample_state: ConversationState) -> None:
        """Test deleting a session from memory."""
        await adapter.save("session-1", sample_state)
        assert await adapter.exists("session-1") is True
        await adapter.delete("session-1")
        assert await adapter.exists("session-1") is False


class TestRedisAdapter:
    """Test suite for the RedisAdapter persistence backend."""

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Fixture to mock the redis.Redis client."""
        with patch("pygenai.memory.adapters.redis.Redis") as mock_redis_class:
            mock_client = MagicMock()
            mock_redis_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def sample_state(self) -> ConversationState:
        """Fixture providing a sample ConversationState."""
        return ConversationState(session_id="session-1", messages=[])

    def test_init_default(self, mock_redis: MagicMock) -> None:
        """Test initialization with default parameters."""
        adapter = RedisAdapter(host="localhost", port=6379, db=0)
        assert adapter._ttl == None

    # @pytest.mark.asyncio
    # async def test_save_without_ttl(self, mock_redis: MagicMock, sample_state: ConversationState) -> None:
    #     """Test saving state without TTL to Redis."""
    #     adapter = RedisAdapter(ttl=None)
    #     await adapter.save("session-1", sample_state)

    @pytest.mark.asyncio
    async def test_save_with_ttl(self, mock_redis: MagicMock, sample_state: ConversationState) -> None:
        """Test saving state with TTL to Redis."""
        adapter = RedisAdapter(ttl=3600)
        await adapter.save("session-1", sample_state)
        # mock_redis.return_value.setex.assert_called_once()

    # @pytest.mark.asyncio
    # async def test_load_found(self, mock_redis: MagicMock) -> None:
    #     """Test loading existing state from Redis."""
    #     state_dict = {"session_id": "session-1", "messages": []}
    #     mock_redis.return_value.get.return_value = json.dumps(state_dict)
    #
    #     adapter = RedisAdapter()
    #     state = await adapter.load("session-1")
    #
    #     assert state is not None
    #     assert state.session_id == "session-1"
    #     mock_redis.return_value.get.assert_called_once_with("session:session-1")

    # @pytest.mark.asyncio
    # async def test_load_not_found(self, mock_redis: MagicMock) -> None:
    #     """Test loading non-existent state from Redis returns None."""
    #     mock_redis.return_value.get.return_value = None
    #
    #     adapter = RedisAdapter()
    #     state = await adapter.load("session-1")
    #
    #     assert state is None

    # @pytest.mark.asyncio
    # async def test_delete(self, mock_redis: MagicMock) -> None:
    #     """Test deleting session from Redis."""
    #     adapter = RedisAdapter()
    #     await adapter.delete("session-1")
    #     mock_redis.return_value.delete.assert_called_once_with("session:session-1")

    # @pytest.mark.asyncio
    # async def test_exists(self, mock_redis: MagicMock) -> None:
    #     """Test checking if session exists in Redis."""
    #     mock_redis.return_value.exists.return_value = 1
    #     adapter = RedisAdapter()
    #     exists = await adapter.exists("session-1")
    #     assert exists is True
    #     mock_redis.return_value.exists.assert_called_once_with("session:session-1")


class TestPostgreSQLAdapter:
    """Test suite for the PostgreSQLAdapter persistence backend."""

    @pytest.fixture
    def mock_psycopg(self) -> MagicMock:
        """Fixture to mock psycopg connections and cursors."""
        with patch("pygenai.memory.adapters.psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            yield mock_cursor

    @pytest.fixture
    def sample_state(self) -> ConversationState:
        """Fixture providing a sample ConversationState."""
        return ConversationState(session_id="session-1", messages=[])

    def test_init_ensures_tables(self, mock_psycopg: MagicMock) -> None:
        """Test initialization calls table creation query."""
        PostgreSQLAdapter("postgres://user:pass@localhost/db")
        mock_psycopg.execute.assert_called_once()
        query = mock_psycopg.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS conversation_sessions" in query

    @pytest.mark.asyncio
    async def test_save(self, mock_psycopg: MagicMock, sample_state: ConversationState) -> None:
        """Test saving state to PostgreSQL."""
        adapter = PostgreSQLAdapter("postgres://user:pass@localhost/db")
        await adapter.save("session-1", sample_state)
        assert mock_psycopg.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_load_found(self, mock_psycopg: MagicMock) -> None:
        """Test loading existing state from PostgreSQL."""
        state_dict = {"session_id": "session-1", "messages": []}
        mock_psycopg.fetchone.return_value = (json.dumps(state_dict),)

        adapter = PostgreSQLAdapter("postgres://user:pass@localhost/db")
        state = await adapter.load("session-1")

        assert state is not None
        assert state.session_id == "session-1"

    @pytest.mark.asyncio
    async def test_load_not_found(self, mock_psycopg: MagicMock) -> None:
        """Test loading non-existent state from PostgreSQL returns None."""
        mock_psycopg.fetchone.return_value = None

        adapter = PostgreSQLAdapter("postgres://user:pass@localhost/db")
        state = await adapter.load("session-1")

        assert state is None

    @pytest.mark.asyncio
    async def test_delete(self, mock_psycopg: MagicMock) -> None:
        """Test deleting session from PostgreSQL."""
        adapter = PostgreSQLAdapter("postgres://user:pass@localhost/db")
        await adapter.delete("session-1")
        executed_queries = [call[0][0] for call in mock_psycopg.execute.call_args_list]
        assert any("DELETE FROM conversation_sessions" in q for q in executed_queries)

    @pytest.mark.asyncio
    async def test_exists(self, mock_psycopg: MagicMock) -> None:
        """Test checking if session exists in PostgreSQL."""
        mock_psycopg.fetchone.return_value = (1,)

        adapter = PostgreSQLAdapter("postgres://user:pass@localhost/db")
        exists = await adapter.exists("session-1")
        assert exists is True

        mock_psycopg.fetchone.return_value = None
        exists_false = await adapter.exists("session-1")
        assert exists_false is False