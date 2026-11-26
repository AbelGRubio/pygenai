"""Tests for MCP lifecycle and JWT utilities."""

from types import SimpleNamespace

import pytest

from pygenai.agent.utils import jwt_utils, mcp_manager


@pytest.mark.asyncio
async def test_get_jwt_token(monkeypatch) -> None:
    settings = SimpleNamespace(
        client_id="client",
        client_secret="secret",
        jwt_url="https://auth.example/token",
    )
    monkeypatch.setattr(jwt_utils, "get_settings", lambda: settings)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"access_token": "jwt-token"}

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            assert url == "https://auth.example/token"
            assert kwargs["auth"] == ("client", "secret")
            return FakeResponse()

    monkeypatch.setattr(jwt_utils.httpx2, "AsyncClient", lambda: FakeClient())

    token = await jwt_utils.get_jwt_token()

    assert token == "jwt-token"


@pytest.mark.asyncio
async def test_get_jwt_token_requires_credentials(monkeypatch) -> None:
    monkeypatch.setattr(jwt_utils, "get_settings", lambda: SimpleNamespace(client_id=None, client_secret=None))

    with pytest.raises(ValueError, match="CLIENT_ID and CLIENT_SECRET"):
        await jwt_utils.get_jwt_token()


@pytest.mark.asyncio
async def test_get_jwt_token_requires_access_token(monkeypatch) -> None:
    settings = SimpleNamespace(client_id="client", client_secret="secret", jwt_url="https://auth.example/token")
    monkeypatch.setattr(jwt_utils, "get_settings", lambda: settings)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {}

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> FakeResponse:
            assert url == "https://auth.example/token"
            return FakeResponse()

    monkeypatch.setattr(jwt_utils.httpx2, "AsyncClient", lambda: FakeClient())

    with pytest.raises(ValueError, match="access_token"):
        await jwt_utils.get_jwt_token()


def test_mcp_manager_hash_is_stable() -> None:
    manager = mcp_manager.MCPManager()

    first = manager._get_hash({"server": {"command": "node", "args": ["a"]}})
    second = manager._get_hash({"server": {"command": "node", "args": ["a"]}})

    assert first == second
    assert len(first) == 128


@pytest.mark.asyncio
async def test_mcp_manager_caches_tools(monkeypatch) -> None:
    manager = mcp_manager.MCPManager()

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def get_tools(self) -> list[object]:
            self.calls += 1
            return [SimpleNamespace(name="search")]

    client = FakeClient()
    config = {"server": {"transport": "stdio"}}

    first = await manager.get_active_tools(config, client)
    second = await manager.get_active_tools(config, client)

    assert first == second
    assert client.calls == 1


@pytest.mark.asyncio
async def test_mcp_manager_get_session() -> None:
    manager = mcp_manager.MCPManager()

    class FakeClient:
        async def session(self, name: str) -> object:
            return SimpleNamespace(name=name)

    client = FakeClient()

    first = await manager.get_session(client, "server")
    second = await manager.get_session(client, "server")

    assert first is second


def test_get_mcp_manager_is_singleton() -> None:
    mcp_manager.get_mcp_manager.cache_clear()

    first = mcp_manager.get_mcp_manager()
    second = mcp_manager.get_mcp_manager()

    assert first is second
