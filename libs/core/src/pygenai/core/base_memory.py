"""Base interfaces for memory and persistence layers."""

from abc import ABC, abstractmethod
from typing import TypeAlias

from .models import ConversationState, Message, SummaryRecord


class MemoryStore(ABC):
    """Abstract base class for memory storage implementations.

    Defines the interface for storing and retrieving messages and summaries
    within a conversation session.
    """

    @abstractmethod
    async def add_message(self, session_id: str, message: Message) -> None:
        """Add a message to the conversation.

        Args:
            session_id: The session identifier.
            message: The message to add.
        """

    async def update_messages(self, session_id: str, messages: list[Message]) -> None:
        """Update a message in the session.

        Args:
            session_id: The session identifier.
            messages: The list of messages to update.
        """
        return NotImplemented

    @abstractmethod
    async def get_messages(self, session_id: str) -> list[Message]:
        """Retrieve all active messages in the session.

        Args:
            session_id: The session identifier.

        Returns:
            List of messages in the session.
        """

    @abstractmethod
    async def get_state(self, session_id: str) -> ConversationState:
        """Retrieve the complete conversation state.

        Args:
            session_id: The session identifier.

        Returns:
            The complete conversation state including messages and summaries.
        """

    @abstractmethod
    async def add_summary(self, session_id: str, summary: SummaryRecord) -> None:
        """Add a summary record to the session.

        Args:
            session_id: The session identifier.
            summary: The summary record to add.
        """

    async def get_summary(self, session_id: str) -> None:
        """Retrieve the summary for the session.

        Args:
            session_id: The session identifier.
            summary: The summary record to add.
        """
        return NotImplemented

    @abstractmethod
    async def clear_messages(self, session_id: str) -> None:
        """Clear all messages from the session.

        Args:
            session_id: The session identifier.
        """

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete the entire session.

        Args:
            session_id: The session identifier.
        """


class PersistenceAdapter(ABC):
    """Abstract base class for persistence adapters.

    Handles low-level storage and retrieval of conversation state.
    """

    @abstractmethod
    async def save(self, session_id: str, state: ConversationState) -> None:
        """Save the conversation state to persistent storage.

        Args:
            session_id: The session identifier.
            state: The conversation state to save.
        """

    @abstractmethod
    async def load(self, session_id: str) -> ConversationState | None:
        """Load the conversation state from persistent storage.

        Args:
            session_id: The session identifier.

        Returns:
            The conversation state or None if not found.
        """

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete the session from persistent storage.

        Args:
            session_id: The session identifier.
        """

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if a session exists in persistent storage.

        Args:
            session_id: The session identifier.

        Returns:
            True if the session exists, False otherwise.
        """


class SemanticSearchAdapter(ABC):
    """Abstract base class for semantic search adapters.

    Enables similarity-based retrieval of past interactions.
    """

    @abstractmethod
    async def index_message(self, session_id: str, message: Message) -> None:
        """Index a message for semantic search.

        Args:
            session_id: The session identifier.
            message: The message to index.
        """

    @abstractmethod
    async def search(self, session_id: str, query: str, top_k: int = 5) -> list[Message]:
        """Search for semantically similar messages.

        Args:
            session_id: The session identifier.
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            List of semantically similar messages.
        """

    @abstractmethod
    async def clear_index(self, session_id: str) -> None:
        """Clear the search index for a session.

        Args:
            session_id: The session identifier.
        """


MemoryTypes: TypeAlias = MemoryStore | PersistenceAdapter | SemanticSearchAdapter  # noqa: UP040
