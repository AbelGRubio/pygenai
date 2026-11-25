"""Tests for agent settings and model definitions."""

from pygenai.agent.models import AgentState, LLMConfig
from pygenai.agent.settings import AgentSettings, get_settings


def test_agent_settings_defaults_and_aliases(monkeypatch) -> None:
    monkeypatch.setenv("JWT_PROTECTED", "false")
    monkeypatch.setenv("JWT_URL", "https://auth.example/token")
    monkeypatch.setenv("CLIENT_ID", "client")
    monkeypatch.setenv("CLIENT_SECRET", "secret")

    settings = AgentSettings()

    assert settings.jwt_protected is False
    assert settings.jwt_url == "https://auth.example/token"
    assert settings.client_id == "client"
    assert settings.client_secret == "secret"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second


def test_llm_config_defaults() -> None:
    config = LLMConfig(provider="openai", model_name="gpt-4o-mini")

    assert config.temperature == 0.7
    assert config.retry_config == {"max_retries": 2}
    assert config.api_base is None


def test_agent_state_accepts_expected_fields() -> None:
    state: AgentState = {
        "messages": [],
        "context": {},
        "rag_query": None,
        "rag_texts": [],
        "tools": [],
        "memory_data": {},
        "session_id": None,
    }

    assert state["messages"] == []
