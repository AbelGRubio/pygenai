"""Example: Using SummarizedMemory with automatic compression."""

import asyncio

from pygenai.memory import SummarizedMemory
from pygenai.core.base_memory import Message
from pygenai.core.models import MessageRole

async def main():
    """Demonstrate SummarizedMemory auto-compression."""
    print("=== SummarizedMemory Example ===\n")

    # Create summarized memory that triggers compression at 80% of 500 tokens
    memory = SummarizedMemory(max_tokens=500, summary_trigger_ratio=0.8)
    session_id = "summarized-example-session"

    print("Trigger threshold: 500 * 0.8 = 400 tokens\n")
    print("Adding messages with 60 tokens each (compression happens at 400+ tokens)\n")

    # Add messages that will trigger compression
    for i in range(12):
        message = Message(
            role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}: " + ("Hello " * 10),
            tokens=60,
        )
        await memory.add_message(session_id, message)

        state = await memory.get_state(session_id)

        if state.summaries:
            print(f"Message {i + 1}:")
            print(f"  Active messages: {len(state.messages)}")
            print(f"  Total tokens in active: {sum(m.tokens for m in state.messages)}")
            print(f"  Summaries created: {len(state.summaries)}")
            if state.summaries:
                print(f"  Last summary: {state.summaries[-1].summary[:60]}...")
            print()

    # Display final state
    final_state = await memory.get_state(session_id)
    print("\nFinal State:")
    print(f"  Active messages: {len(final_state.messages)}")
    print(f"  Total tokens (active): {sum(m.tokens for m in final_state.messages)}")
    print(f"  Total summaries: {len(final_state.summaries)}")

    print("\nAll Summaries:")
    for i, summary in enumerate(final_state.summaries, 1):
        print(f"\nSummary {i}:")
        print(f"  Messages compressed: {summary.messages_count}")
        print(f"  Original tokens: {summary.original_tokens}")
        print(f"  Content: {summary.summary[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
