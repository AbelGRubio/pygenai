"""Example: Using WindowMemory to maintain a sliding context window."""

import asyncio

from pygenai.memory import WindowMemory
from pygenai.core.base_memory import Message
from pygenai.core.models import MessageRole

async def main():
    """Demonstrate WindowMemory usage with token and turn limits."""
    print("=== WindowMemory Example ===\n")

    # Create a window memory with 1000 token limit and 5 turn limit
    memory = WindowMemory(max_tokens=1000, max_turns=5)
    session_id = "window-example-session"

    print("Adding 15 messages with 150 tokens each...")
    print("(Window will trim to keep within 1000 tokens and 5 turns)\n")

    # Add many messages
    for i in range(15):
        message = Message(
            role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message #{i}: " + "Lorem ipsum dolor sit amet, " * 5,
            tokens=150,
        )
        await memory.add_message(session_id, message)

        if i % 5 == 4:  # Print status every 5 messages
            state = await memory.get_state(session_id)
            print(f"After message {i + 1}:")
            print(f"  Messages in window: {len(state.messages)}")
            print(f"  Total tokens: {state.total_tokens}")
            print()

    # Display final state
    final_state = await memory.get_state(session_id)
    print("\nFinal Window State:")
    print(f"  Messages in window: {len(final_state.messages)}")
    print(f"  Total tokens: {final_state.total_tokens}")

    # Show messages in window
    print("\nMessages in window:")
    for msg in final_state.messages[-3:]:  # Show last 3
        preview = msg.content[:50]
        print(f"  - {msg.type}: {preview}...")


if __name__ == "__main__":
    asyncio.run(main())
