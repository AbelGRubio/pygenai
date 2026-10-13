"""Example: Using persistence adapters to store conversation state."""

import asyncio

from pygenai.memory import (
    ConversationState,
    InMemoryAdapter,
    Message,
    MessageRole,
)


async def main():
    """Demonstrate persistence adapter usage."""
    print("=== Persistence Adapter Example ===\n")

    # Create an in-memory adapter
    adapter = InMemoryAdapter()
    session_id = "persistent-session-123"

    # Create a conversation state
    print("Creating conversation state...\n")
    state = ConversationState(
        session_id=session_id,
        metadata={"user_id": "user_456", "model": "gpt-4"},
    )

    # Add messages
    messages = [
        Message(role=MessageRole.USER, content="Hello, how can you help?", tokens=8),
        Message(role=MessageRole.ASSISTANT, content="I can help with various tasks!", tokens=9),
        Message(role=MessageRole.USER, content="Great! Let's start.", tokens=5),
    ]

    for msg in messages:
        state.messages.append(msg)
        state.total_tokens += msg.tokens

    print("Initial state:")
    print(f"  Messages: {len(state.messages)}")
    print(f"  Total tokens: {state.total_tokens}")
    print(f"  Metadata: {state.metadata}\n")

    # Save state using adapter
    print("Saving state to adapter...")
    await adapter.save(session_id, state)

    # Check if session exists
    exists = await adapter.exists(session_id)
    print(f"Session exists: {exists}\n")

    # Load state from adapter
    print("Loading state from adapter...\n")
    loaded_state = await adapter.load(session_id)

    if loaded_state:
        print("Loaded state:")
        print(f"  Session ID: {loaded_state.session_id}")
        print(f"  Messages: {len(loaded_state.messages)}")
        print(f"  Total tokens: {loaded_state.total_tokens}")
        print(f"  Metadata: {loaded_state.metadata}")

        print("\nMessages in loaded state:")
        for msg in loaded_state.messages:
            print(f"  - {msg.role.value}: {msg.content}")

    # Delete session
    print("\nDeleting session...")
    await adapter.delete(session_id)

    # Verify deletion
    exists_after = await adapter.exists(session_id)
    print(f"Session exists after deletion: {exists_after}")


if __name__ == "__main__":
    asyncio.run(main())
