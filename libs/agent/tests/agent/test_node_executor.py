"""Tests for agent node execution behavior."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from pygenai.agent.node_executor import AgentNodeExecutor


class FakeLLM:
    def __init__(self, response: str = "YES: python") -> None:
        self.response = response

    async def ainvoke(self, messages: object) -> SimpleNamespace:
        return SimpleNamespace(content=self.response)


class FakeRetriever:
    def __init__(self, result: str = "retrieved text") -> None:
        self.result = result

    def retrieve_and_format(self, query: str) -> str:
        return f"{self.result}: {query}"


@pytest.mark.asyncio
async def test_prepare_rag_query_no_sources(monkeypatch) -> None:
    executor = AgentNodeExecutor(FakeLLM(), {}, None, [], {})

    result = await executor.prepare_rag_query_node({}, {"configurable": {"thread_id": "session-1"}})

    assert result == {"session_id": "session-1"}


@pytest.mark.asyncio
async def test_prepare_rag_query_returns_retrieved_text(monkeypatch) -> None:
    executor = AgentNodeExecutor(FakeLLM(), {"docs": FakeRetriever()}, None, [], {})

    state = {"messages": [HumanMessage(content="What is Python?")]}
    result = await executor.prepare_rag_query_node(state, {"configurable": {}})

    assert result["rag_query"] == "PYTHON"
    assert result["rag_texts"] == ["--- Content from RAG docs [retrieved text: PYTHON]"]
    assert result["session_id"]


@pytest.mark.asyncio
async def test_prepare_rag_query_handles_llm_error(monkeypatch) -> None:
    class BrokenLLM:
        async def ainvoke(self, messages: object) -> object:
            raise RuntimeError("llm unavailable")

    executor = AgentNodeExecutor(BrokenLLM(), {"docs": FakeRetriever()}, None, [], {})
    state = {"messages": [HumanMessage(content="Question")]}

    result = await executor.prepare_rag_query_node(state, {"configurable": {"thread_id": "session-1"}})

    assert result["rag_query"] == ""
    assert result["rag_texts"] == ["--- Content from RAG docs [retrieved text: ]"]


@pytest.mark.asyncio
async def test_prepare_tools_combines_skills(monkeypatch) -> None:
    skill = SimpleNamespace(name="skill-tool")
    executor = AgentNodeExecutor(FakeLLM(), {}, None, [skill], {})

    result = await executor.prepare_tools_node({})

    assert result == {"tools": ["skill-tool"]}


@pytest.mark.asyncio
async def test_memory_node_returns_context(monkeypatch) -> None:
    class FakeMemory:
        async def get_messages(self, session_id: str) -> list[str]:
            assert session_id == "session-1"
            return ["old message"]

    executor = AgentNodeExecutor(FakeLLM(), {}, FakeMemory(), [], {})

    result = await executor.memory_node({"context": {}, "session_id": "session-1"})

    assert result == {"context": ["old message"], "memory_data": ["old message"]}


@pytest.mark.asyncio
async def test_memory_node_handles_error(monkeypatch) -> None:
    class BrokenMemory:
        async def get_messages(self, session_id: str) -> list[str]:
            raise RuntimeError("memory unavailable")

    executor = AgentNodeExecutor(FakeLLM(), {}, BrokenMemory(), [], {})

    result = await executor.memory_node({"context": {"existing": True}, "session_id": "session-1"})

    assert result["context"]["memory_error"] == "memory unavailable"


@pytest.mark.asyncio
async def test_save_memory_without_memory() -> None:
    executor = AgentNodeExecutor(FakeLLM(), {}, None, [], {})

    assert await executor.save_memory_node({"session_id": "session-1", "messages": []}) == {}


@pytest.mark.asyncio
async def test_save_memory_updates_messages(monkeypatch) -> None:
    class FakeMemory:
        def __init__(self) -> None:
            self.updated: tuple[str, list[object]] | None = None

        async def update_messages(self, session_id: str, messages: list[object]) -> None:
            self.updated = (session_id, messages)

    memory = FakeMemory()
    executor = AgentNodeExecutor(FakeLLM(), {}, memory, [], {})
    messages = [HumanMessage(content="hello")]

    result = await executor.save_memory_node({"session_id": "session-1", "messages": messages})

    assert result == {}
    assert memory.updated == ("session-1", messages)


@pytest.mark.asyncio
async def test_prepare_tools_loads_mcp_tools(monkeypatch) -> None:
    executor = AgentNodeExecutor(FakeLLM(), {}, None, [], {"server": {"transport": "stdio"}})

    async def fake_start() -> None:
        executor._mcp_client = SimpleNamespace()

    async def fake_tools(config: object, client: object) -> list[object]:
        return [SimpleNamespace(name="mcp-tool")]

    monkeypatch.setattr(executor, "_start_mcp_client", fake_start)
    monkeypatch.setattr(executor._mcp_manager, "get_active_tools", fake_tools)

    result = await executor.prepare_tools_node({})

    assert result == {"tools": ["mcp-tool"]}


@pytest.mark.asyncio
async def test_prepare_tools_handles_mcp_error(monkeypatch) -> None:
    executor = AgentNodeExecutor(FakeLLM(), {}, None, [], {"server": {"transport": "stdio"}})

    async def broken_start() -> None:
        raise RuntimeError("mcp unavailable")

    monkeypatch.setattr(executor, "_start_mcp_client", broken_start)

    result = await executor.prepare_tools_node({})

    assert result == {"tools": []}


@pytest.mark.asyncio
async def test_reasoning_node_calls_react_agent(monkeypatch) -> None:
    executor = AgentNodeExecutor(FakeLLM(), {}, None, [], {})

    class FakeReactAgent:
        async def ainvoke(self, payload: dict[str, object]) -> dict[str, list[object]]:
            assert payload["messages"]
            return {"messages": [HumanMessage(content="answer")]}

    monkeypatch.setattr(
        "pygenai.agent.node_executor.create_agent",
        lambda model, tools, system_prompt: FakeReactAgent(),
    )

    result = await executor.reasoning_node({"messages": [HumanMessage(content="question")], "rag_texts": [], "context": [], "memory_data": None})

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "answer"


@pytest.mark.asyncio
async def test_save_memory_creates_summary(monkeypatch) -> None:
    class FakeMemory:
        async def update_messages(self, session_id: str, messages: list[object]) -> None:
            return None

        async def get_summary(self, session_id: str) -> str:
            return "existing summary"

        async def add_summary(self, session_id: str, summary: str) -> None:
            assert summary == "new summary"

    llm = FakeLLM("new summary")
    executor = AgentNodeExecutor(llm, {}, FakeMemory(), [], {})
    messages = [HumanMessage(content=f"message-{i}") for i in range(7)]

    result = await executor.save_memory_node({"session_id": "session-1", "messages": messages})

    assert result == {}


@pytest.mark.asyncio
async def test_save_memory_handles_summary_failure(monkeypatch) -> None:
    class FakeMemory:
        async def update_messages(self, session_id: str, messages: list[object]) -> None:
            return None

        async def get_summary(self, session_id: str) -> str:
            raise RuntimeError("summary unavailable")

    executor = AgentNodeExecutor(FakeLLM(), {}, FakeMemory(), [], {})

    result = await executor.save_memory_node({"session_id": "session-1", "messages": [HumanMessage(content=str(i)) for i in range(7)]})

    assert result == {}


@pytest.mark.asyncio
async def test_close_without_mcp_client() -> None:
    executor = AgentNodeExecutor(FakeLLM(), {}, None, [], {})

    await executor.close()


@pytest.mark.asyncio
async def test_close_mcp_client(monkeypatch) -> None:
    executor = AgentNodeExecutor(FakeLLM(), {}, None, [], {})

    class FakeClient:
        async def close(self) -> None:
            self.closed = True

    client = FakeClient()
    executor._mcp_client = client

    await executor.close()

    assert client.closed is True
