"""Tests for agent builder and declarative base definition."""

from types import SimpleNamespace

from pygenai.agent.base_definition import BaseAgentDefinition
from pygenai.agent.builder import AgentBuilder
from pygenai.agent.models import LLMConfig


class FakeBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def with_llm(self, config: object) -> "FakeBuilder":
        self.calls.append(("llm", config))
        return self

    def add_memory(self, memory: object) -> "FakeBuilder":
        self.calls.append(("memory", memory))
        return self

    def add_rag_source(self, name: str, retriever: object) -> "FakeBuilder":
        self.calls.append(("rag", (name, retriever)))
        return self

    def add_mcp(self, name: str, mcp: object) -> "FakeBuilder":
        self.calls.append(("mcp", (name, mcp)))
        return self

    def build(self) -> object:
        self.calls.append(("build", None))
        return object()


class CompleteDefinition(BaseAgentDefinition):
    def configure_llm(self) -> LLMConfig:
        return LLMConfig(provider="openai", model_name="test-model")

    def configure_memory(self) -> object:
        return SimpleNamespace()

    def configure_rag(self) -> dict[str, object]:
        return {"docs": SimpleNamespace()}

    def configure_mcps(self) -> dict[str, object]:
        return {"server": SimpleNamespace()}


def test_base_definition_assembles_all_features(monkeypatch) -> None:
    builders: list[FakeBuilder] = []

    def factory() -> FakeBuilder:
        builder = FakeBuilder()
        builders.append(builder)
        return builder

    definition = CompleteDefinition(builder=factory)

    graph = definition.create_graph()

    assert graph is not None
    assert builders[0].calls[0][0] == "llm"
    assert ("rag", ("docs", builders[0].calls[2][1][1])) in builders[0].calls
    assert ("mcp", ("server", builders[0].calls[3][1][1])) in builders[0].calls
    assert builders[0].calls[-1] == ("build", None)


def test_builder_adders_are_fluent(monkeypatch) -> None:
    llm = SimpleNamespace()
    monkeypatch.setattr("pygenai.agent.builder.ChatLiteLLM", lambda **kwargs: llm)
    monkeypatch.setattr("pygenai.agent.builder.get_settings", lambda: SimpleNamespace(llm_api_key=SimpleNamespace(get_secret_value=lambda: "key"), model_base_url="http://llm"))

    retriever = object()
    memory = object()
    skill = object()

    builder = AgentBuilder()
    result = builder.add_rag_source("docs", retriever).add_memory(memory).add_skill(skill).add_mcp("server", {"command": "server"})

    assert result is builder
    assert builder._rag_sources == {"docs": retriever}
    assert builder._memory is memory
    assert builder._skills == [skill]
    assert builder._mcps == {"server": {"command": "server"}}


def test_builder_with_llm_updates_model(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat_lite_llm(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("pygenai.agent.builder.ChatLiteLLM", fake_chat_lite_llm)
    monkeypatch.setattr(
        "pygenai.agent.builder.get_settings",
        lambda: SimpleNamespace(
            llm_api_key=SimpleNamespace(get_secret_value=lambda: "secret"),
            model_base_url="http://default",
        ),
    )

    builder = AgentBuilder()
    result = builder.with_llm(LLMConfig(provider="openai", model_name="custom", api_base="http://custom", temperature=0.2))

    assert result is builder
    assert captured["model"] == "custom"
    assert captured["temperature"] == 0.2
    assert captured["api_base"] == "http://custom"
