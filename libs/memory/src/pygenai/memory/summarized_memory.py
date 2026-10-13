"""SummarizedMemory: automatically compresses old messages into summaries."""

from collections.abc import Callable

from pygenai.core.base_memory import MemoryStore
from pygenai.core.models import ConversationState, Message, MessageRole, SummaryRecord

MIN_MESSAGES_TO_COMPRESS = 3


class SummarizedMemory(MemoryStore):
    """
    Memory store that auto-compresses old messages into summaries.

    When conversation tokens exceed a threshold, automatically summarizes
    the oldest messages while keeping recent ones intact for context.
    """

    def __init__(
        self,
        max_tokens: int = 8192,
        summary_trigger_ratio: float = 0.8,
        summarizer: Callable[[list[Message]], str] | None = None,
    ):
        """
        Initialize SummarizedMemory.

        Args:
            max_tokens: Maximum tokens before triggering compression (default: 8192).
            summary_trigger_ratio: Ratio of max_tokens that triggers summarization
                (default: 0.8, i.e., summarize at 80% capacity).
            summarizer: Optional callable to summarize messages. If None, uses default.
        """
        self._max_tokens = max_tokens
        self._trigger_tokens = int(max_tokens * summary_trigger_ratio)
        self._summarizer = summarizer or self._default_summarizer
        self._sessions: dict[str, ConversationState] = {}

    async def add_message(self, session_id: str, message: Message) -> None:
        """
        Add a message and trigger summarization if needed.

        Args:
            session_id: The session identifier.
            message: The message to add.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)

        state = self._sessions[session_id]
        state.messages.append(message)
        state.total_tokens += message.tokens

        await self._compress_if_needed(session_id)

    async def get_messages(self, session_id: str) -> list[Message]:
        """
        Retrieve active messages (excluding summaries).

        Args:
            session_id: The session identifier.

        Returns:
            List of active messages.
        """
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id].messages

    async def get_state(self, session_id: str) -> ConversationState:
        """
        Retrieve the complete state including summaries.

        Args:
            session_id: The session identifier.

        Returns:
            The conversation state with messages and summaries.
        """
        if session_id not in self._sessions:
            return ConversationState(session_id=session_id)
        return self._sessions[session_id]

    async def add_summary(self, session_id: str, summary: SummaryRecord) -> None:
        """
        Add a summary record.

        Args:
            session_id: The session identifier.
            summary: The summary record.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)
        self._sessions[session_id].summaries.append(summary)

    async def clear_messages(self, session_id: str) -> None:
        """
        Clear all active messages (keep summaries).

        Args:
            session_id: The session identifier.
        """
        if session_id in self._sessions:
            self._sessions[session_id].messages.clear()
            self._sessions[session_id].total_tokens = 0

    async def delete_session(self, session_id: str) -> None:
        """
        Delete the entire session.

        Args:
            session_id: The session identifier.
        """
        self._sessions.pop(session_id, None)

    async def _compress_if_needed(self, session_id: str) -> None:
        """
        Check if compression is needed and summarize old messages.

        Args:
            session_id: The session identifier.
        """
        state = self._sessions[session_id]

        if state.total_tokens < self._trigger_tokens:
            return

        if len(state.messages) < MIN_MESSAGES_TO_COMPRESS:
            return

        num_to_summarize = max(1, len(state.messages) // 2)
        messages_to_summarize = state.messages[:num_to_summarize]

        if messages_to_summarize:
            summary_text = self._summarizer(messages_to_summarize)
            summary = SummaryRecord(
                summary=summary_text,
                messages_count=len(messages_to_summarize),
                original_tokens=sum(m.tokens for m in messages_to_summarize),
                metadata={
                    "roles": {m.type for m in messages_to_summarize},
                },
            )

            state.summaries.append(summary)
            removed_tokens = sum(m.tokens for m in messages_to_summarize)
            state.messages = state.messages[num_to_summarize:]
            state.total_tokens -= removed_tokens

    @staticmethod
    def _default_summarizer(messages: list[Message]) -> str:
        """
        Default summarizer: creates a brief recap of messages.

        Args:
            messages: Messages to summarize.

        Returns:
            Summary text.
        """
        if not messages:
            return "No messages to summarize."

        user_msgs = [m for m in messages if m.type == MessageRole.USER]
        assistant_msgs = [m for m in messages if m.type == MessageRole.ASSISTANT]

        summary_lines = [
            f"Summary of {len(messages)} messages:",
            f"- User queries: {len(user_msgs)}",
            f"- Assistant responses: {len(assistant_msgs)}",
        ]

        if user_msgs:
            first_user = user_msgs[0].content[:100]
            summary_lines.append(f"- First user message: {first_user}...")

        if assistant_msgs:
            first_assistant = assistant_msgs[0].content[:100]
            summary_lines.append(f"- First response: {first_assistant}...")

        return "\n".join(summary_lines)

    def set_summarizer(self, summarizer: Callable[[list[Message]], str]) -> None:
        """
        Set a custom summarizer function.

        Args:
            summarizer: Function that takes a list of Messages and returns a string.
        """
        self._summarizer = summarizer
