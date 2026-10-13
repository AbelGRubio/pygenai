"""BufferMemory: stores complete conversation history sequentially."""

from pygenai.core.base_memory import MemoryStore
from pygenai.core.models import ConversationState, Message


class BufferMemory(MemoryStore):
    """
    Simple memory store that keeps all messages from the session.

    Stores the complete conversation history sequentially without compression.
    Suitable for short conversations or when full history is required.
    """

    def __init__(self):
        """Initialize BufferMemory."""
        self._sessions: dict[str, ConversationState] = {}

    async def add_message(self, session_id: str, message: Message) -> None:
        """
        Add a message to the session.

        Args:
            session_id: The session identifier.
            message: The message to add.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)

        state = self._sessions[session_id]
        state.messages.append(message)
        # state.total_tokens += message.tokens

    async def update_messages(self, session_id: str, messages: list[Message]) -> None:
        """
        Update a message in the session.

        Args:
            session_id: The session identifier.
            messages: The list of messages to update.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)

        state = self._sessions[session_id]
        state.messages = messages

    async def get_messages(self, session_id: str) -> list[Message]:
        """
        Retrieve all messages in the session.

        Args:
            session_id: The session identifier.

        Returns:
            List of all messages.
        """
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id].messages

    async def get_state(self, session_id: str) -> ConversationState:
        """
        Retrieve the complete conversation state.

        Args:
            session_id: The session identifier.

        Returns:
            The conversation state.
        """
        if session_id not in self._sessions:
            return ConversationState(session_id=session_id)
        return self._sessions[session_id]

    async def add_summary(self, session_id, summary) -> None:
        """
        Add a summary record (not typically used in BufferMemory).

        Args:
            session_id: The session identifier.
            summary: The summary record.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)
        self._sessions[session_id].summaries.append(summary)

    async def get_summary(self, session_id) -> str:
        """
        Add a summary record (not typically used in BufferMemory).

        Args:
            session_id: The session identifier.
        """
        if session_id not in self._sessions:
            return ""
        summaries_ = [s.summary for s in self._sessions[session_id].summaries]
        return "\n".join(summaries_)

    async def clear_messages(self, session_id: str) -> None:
        """
        Clear all messages from the session.

        Args:
            session_id: The session identifier.
        """
        if session_id in self._sessions:
            self._sessions[session_id].messages.clear()
            # self._sessions[session_id].total_tokens = 0

    async def delete_session(self, session_id: str) -> None:
        """
        Delete the entire session.

        Args:
            session_id: The session identifier.
        """
        self._sessions.pop(session_id, None)
