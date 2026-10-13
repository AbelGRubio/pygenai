"""Example: Using BufferMemory for simple sequential conversation history."""

import asyncio

from pygenai.memory import BufferMemory, Message, MessageRole, TokenCounter


async def main():
    """Demonstrate BufferMemory usage."""
    print("=== BufferMemory Example ===\n")

    # Create a buffer memory instance
    memory = BufferMemory()
    session_id = "user-alice-session"

    # Simulate a conversation
    conversation = [
        ("user", "What is machine learning?"),
        ("assistant", "Machine learning is a subset of AI where computers learn from data."),
        ("user", "Can you give me an example?"),
        ("assistant", "Sure! Image recognition is a common example of ML."),
    ]

    # Add messages to memory
    print("Adding messages to memory...\n")
    for role, content in conversation:
        tokens = TokenCounter.count(content)
        message = Message(
            role=MessageRole(role),
            content=content,
            tokens=tokens,
        )
        await memory.add_message(session_id, message)
        print(f"[{role.upper()}]: {content}")
        print(f"  Tokens: {tokens}\n")

    # Retrieve and display state
    state = await memory.get_state(session_id)
    print("\nConversation Summary:")
    print(f"  Total messages: {len(state.messages)}")
    print(f"  Total tokens: {state.total_tokens}")
    print(f"  Session ID: {state.session_id}")


if __name__ == "__main__":
    asyncio.run(main())
