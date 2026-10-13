"""Persistence adapters for different storage backends."""

import json

import psycopg
import redis
from pygenai.core.base_memory import PersistenceAdapter
from pygenai.core.models import ConversationState


class InMemoryAdapter(PersistenceAdapter):
    """
    In-memory persistence adapter fora development and testing.

    Stores conversation state in RAM. Data is lost when the process terminates.
    """

    def __init__(self):
        """Initialize InMemoryAdapter."""
        self._storage: dict[str, ConversationState] = {}

    async def save(self, session_id: str, state: ConversationState) -> None:
        """
        Save state to memory.

        Args:
            session_id: The session identifier.
            state: The conversation state to save.
        """
        self._storage[session_id] = state

    async def load(self, session_id: str) -> ConversationState | None:
        """
        Load state from memory.

        Args:
            session_id: The session identifier.

        Returns:
            The conversation state or None if not found.
        """
        return self._storage.get(session_id)

    async def delete(self, session_id: str) -> None:
        """
        Delete session from memory.

        Args:
            session_id: The session identifier.
        """
        self._storage.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        """
        Check if session exists in memory.

        Args:
            session_id: The session identifier.

        Returns:
            True if session exists.
        """
        return session_id in self._storage


class RedisAdapter(PersistenceAdapter):
    """
    Redis persistence adapter for fast key-value storage.

    Suitable for active sessions in production with automatic expiration.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, ttl: int | None = None):
        """
        Initialize RedisAdapter.

        Args:
            host: Redis host (default: localhost).
            port: Redis port (default: 6379).
            db: Redis database number (default: 0).
            ttl: Optional TTL in seconds for session expiration.
        """
        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._ttl = ttl

    async def save(self, session_id: str, state: ConversationState) -> None:
        """
        Save state to Redis.

        Args:
            session_id: The session identifier.
            state: The conversation state to save.
        """
        key = f"session:{session_id}"
        serialized = json.dumps(state.to_dict())
        if self._ttl:
            self._client.setex(key, self._ttl, serialized)
        else:
            self._client.set(key, serialized)

    async def load(self, session_id: str) -> ConversationState | None:
        """
        Load state from Redis.

        Args:
            session_id: The session identifier.

        Returns:
            The conversation state or None if not found.
        """
        key = f"session:{session_id}"
        data = self._client.get(key)
        if not data:
            return None
        return ConversationState.from_dict(json.loads(data))

    async def delete(self, session_id: str) -> None:
        """
        Delete session from Redis.

        Args:
            session_id: The session identifier.
        """
        key = f"session:{session_id}"
        self._client.delete(key)

    async def exists(self, session_id: str) -> bool:
        """
        Check if session exists in Redis.

        Args:
            session_id: The session identifier.

        Returns:
            True if session exists.
        """
        key = f"session:{session_id}"
        return self._client.exists(key) > 0


class PostgreSQLAdapter(PersistenceAdapter):
    """
    PostgreSQL persistence adapter for relational storage.

    Suitable for long-term storage and complex queries on conversation history.
    """

    def __init__(self, conninfo: str):
        """
        Initialize PostgreSQLAdapter.

        Args:
            conninfo: PostgreSQL connection string.
        """
        self._conninfo = conninfo
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure required tables exist."""
        with psycopg.connect(self._conninfo) as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    async def save(self, session_id: str, state: ConversationState) -> None:
        """
        Save state to PostgreSQL.

        Args:
            session_id: The session identifier.
            state: The conversation state to save.
        """
        with psycopg.connect(self._conninfo) as conn, conn.cursor() as cur:
            data = json.dumps(state.to_dict())
            cur.execute(
                """
                INSERT INTO conversation_sessions (session_id, data)
                VALUES (%s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                data = EXCLUDED.data,
                updated_at = CURRENT_TIMESTAMP
            """,
                (session_id, data),
            )
            conn.commit()

    async def load(self, session_id: str) -> ConversationState | None:
        """
        Load state from PostgreSQL.

        Args:
            session_id: The session identifier.

        Returns:
            The conversation state or None if not found.
        """
        with psycopg.connect(self._conninfo) as conn, conn.cursor() as cur:
            cur.execute("SELECT data FROM conversation_sessions WHERE session_id = %s", (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            return ConversationState.from_dict(json.loads(row[0]))

    async def delete(self, session_id: str) -> None:
        """
        Delete session from PostgreSQL.

        Args:
            session_id: The session identifier.
        """
        with psycopg.connect(self._conninfo) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM conversation_sessions WHERE session_id = %s", (session_id,))
            conn.commit()

    async def exists(self, session_id: str) -> bool:
        """
        Check if session exists in PostgreSQL.

        Args:
            session_id: The session identifier.

        Returns:
            True if session exists.
        """
        with psycopg.connect(self._conninfo) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM conversation_sessions WHERE session_id = %s LIMIT 1", (session_id,))
            return cur.fetchone() is not None
