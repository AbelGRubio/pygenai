"""Tests for memory components."""

import pytest

from pygenai.memory import (
    BufferMemory,
    ConversationState,
    InMemoryAdapter,
    Message,
    MessageRole,
    SummarizedMemory,
    WindowMemory,
)

# Test constants
EXPECTED_WINDOW_SIZE_BY_TOKENS = 3
EXPECTED_WINDOW_SIZE_BY_TURNS = 10
EXPECTED_AUTO_COMPRESS_MESSAGES = 20
EXPECTED_MESSAGES_COUNT = 2
EXPECTED_SINGLE_MESSAGE = 1
EXPECTED_INITIAL_TOKENS = 10


@pytest.fixture
def buffer_memory():
    """Create a BufferMemory instance."""
    return BufferMemory()


@pytest.fixture
def window_memory():
    """Create a WindowMemory instance."""
    return WindowMemory(max_tokens=1000, max_turns=10)


@pytest.fixture
def summarized_memory():
    """Create a SummarizedMemory instance."""
    return SummarizedMemory(max_tokens=500)


@pytest.fixture
def in_memory_adapter():
    """Create an InMemoryAdapter instance."""
    return InMemoryAdapter()


class TestBufferMemory:
    """Tests for BufferMemory."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_messages(self, buffer_memory):
        """Test adding and retrieving messages."""
        session_id = "test-session"
        msg1 = Message(role=MessageRole.USER, content="Hello", tokens=5)
        msg2 = Message(role=MessageRole.ASSISTANT, content="Hi there", tokens=8)

        await buffer_memory.add_message(session_id, msg1)
        await buffer_memory.add_message(session_id, msg2)

        messages = await buffer_memory.get_messages(session_id)
        assert len(messages) == EXPECTED_MESSAGES_COUNT
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there"

    @pytest.mark.asyncio
    async def test_get_state(self, buffer_memory):
        """Test retrieving conversation state."""
        session_id = "test-session"
        msg = Message(role=MessageRole.USER, content="Test", tokens=EXPECTED_INITIAL_TOKENS)

        await buffer_memory.add_message(session_id, msg)
        state = await buffer_memory.get_state(session_id)

        assert state.session_id == session_id
        assert len(state.messages) == EXPECTED_SINGLE_MESSAGE
        assert state.total_tokens == EXPECTED_INITIAL_TOKENS

    @pytest.mark.asyncio
    async def test_clear_messages(self, buffer_memory):
        """Test clearing messages."""
        session_id = "test-session"
        msg = Message(role=MessageRole.USER, content="Test", tokens=10)

        await buffer_memory.add_message(session_id, msg)
        await buffer_memory.clear_messages(session_id)

        messages = await buffer_memory.get_messages(session_id)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_delete_session(self, buffer_memory):
        """Test deleting a session."""
        session_id = "test-session"
        msg = Message(role=MessageRole.USER, content="Test", tokens=10)

        await buffer_memory.add_message(session_id, msg)
        await buffer_memory.delete_session(session_id)

        messages = await buffer_memory.get_messages(session_id)
        assert len(messages) == 0


class TestWindowMemory:
    """Tests for WindowMemory."""

    @pytest.mark.asyncio
    async def test_window_trim_by_tokens(self, window_memory):
        """Test that window trims messages when exceeding token limit."""
        session_id = "test-session"

        for i in range(EXPECTED_AUTO_COMPRESS_MESSAGES):
            msg = Message(
                role=MessageRole.USER if i % EXPECTED_WINDOW_SIZE_BY_TOKENS == 0 else MessageRole.ASSISTANT,
                content=f"Message {i}",
                tokens=150,
            )
            await window_memory.add_message(session_id, msg)

        messages = await window_memory.get_messages(session_id)
        total_tokens = sum(m.tokens for m in messages)
        assert total_tokens <= window_memory._max_tokens

    @pytest.mark.asyncio
    async def test_window_trim_by_turns(self):
        """Test that window trims by turn limit."""
        window_mem = WindowMemory(max_tokens=10000, max_turns=EXPECTED_WINDOW_SIZE_BY_TURNS)
        session_id = "test-session"

        for i in range(EXPECTED_WINDOW_SIZE_BY_TURNS + 5):
            msg = Message(
                role=MessageRole.USER,
                content=f"Message {i}",
                tokens=50,
            )
            await window_mem.add_message(session_id, msg)

        messages = await window_mem.get_messages(session_id)
        assert len(messages) == EXPECTED_WINDOW_SIZE_BY_TURNS


class TestSummarizedMemory:
    """Tests for SummarizedMemory."""

    @pytest.mark.asyncio
    async def test_auto_compression(self, summarized_memory):
        """Test that summarization occurs when threshold is exceeded."""
        session_id = "test-session"

        for i in range(EXPECTED_AUTO_COMPRESS_MESSAGES):
            msg = Message(
                role=MessageRole.USER if i % EXPECTED_WINDOW_SIZE_BY_TOKENS == 0 else MessageRole.ASSISTANT,
                content=f"Message {i}" * 10,
                tokens=100,
            )
            await summarized_memory.add_message(session_id, msg)

        state = await summarized_memory.get_state(session_id)
        assert len(state.summaries) > 0, "Summaries should have been created"
        assert len(state.messages) < EXPECTED_AUTO_COMPRESS_MESSAGES, "Old messages should have been summarized"

    @pytest.mark.asyncio
    async def test_custom_summarizer(self, summarized_memory):
        """Test using a custom summarizer."""

        def custom_summarizer(messages: list[Message]) -> str:
            return f"Custom summary of {len(messages)} messages"

        summarized_memory.set_summarizer(custom_summarizer)
        session_id = "test-session"

        for i in range(EXPECTED_AUTO_COMPRESS_MESSAGES):
            msg = Message(
                role=MessageRole.USER,
                content=f"Message {i}" * 10,
                tokens=100,
            )
            await summarized_memory.add_message(session_id, msg)

        state = await summarized_memory.get_state(session_id)
        if state.summaries:
            assert "Custom summary" in state.summaries[0].summary


class TestInMemoryAdapter:
    """Tests for InMemoryAdapter."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, in_memory_adapter):
        """Test saving and loading state."""
        session_id = "test-session"
        state = ConversationState(session_id=session_id)
        msg = Message(role=MessageRole.USER, content="Test", tokens=10)
        state.messages.append(msg)

        await in_memory_adapter.save(session_id, state)
        loaded = await in_memory_adapter.load(session_id)

        assert loaded is not None
        assert loaded.session_id == session_id
        assert len(loaded.messages) == 1

    @pytest.mark.asyncio
    async def test_delete(self, in_memory_adapter):
        """Test deleting a session."""
        session_id = "test-session"
        state = ConversationState(session_id=session_id)

        await in_memory_adapter.save(session_id, state)
        await in_memory_adapter.delete(session_id)

        loaded = await in_memory_adapter.load(session_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_exists(self, in_memory_adapter):
        """Test checking session existence."""
        session_id = "test-session"
        state = ConversationState(session_id=session_id)

        assert not (await in_memory_adapter.exists(session_id))
        await in_memory_adapter.save(session_id, state)
        assert await in_memory_adapter.exists(session_id)
