"""Tests for authentication middleware."""

from types import SimpleNamespace

import pytest
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from pygenai.core.middlewares.auth import AuthMiddleware


def make_request(path: str = "/private", authorization: str | None = None) -> Request:
    headers = [] if authorization is None else [(b"authorization", authorization.encode())]
    scope = {
        "type": "http",
        "path": path,
        "headers": headers,
        "method": "GET",
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 123),
        "scheme": "http",
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_validate_accepts_bearer_token(monkeypatch) -> None:
    middleware = AuthMiddleware(lambda scope, receive, send: None)
    adapter = SimpleNamespace(
        get_payload=lambda token: {"sub": "user-1", "token": token},
        get_roles=lambda payload: {"admin"},
    )
    monkeypatch.setattr(middleware, "idp_adapter", adapter)

    request = make_request(authorization="Bearer secret-token")
    validated = await middleware.validate(request)

    assert validated.state.user["sub"] == "user-1"
    assert validated.state.roles == {"admin"}


@pytest.mark.asyncio
async def test_validate_rejects_missing_header(monkeypatch) -> None:
    middleware = AuthMiddleware(lambda scope, receive, send: None)

    with pytest.raises(HTTPException) as exc:
        await middleware.validate(make_request())

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_validate_rejects_invalid_token(monkeypatch) -> None:
    middleware = AuthMiddleware(lambda scope, receive, send: None)

    def invalid_payload(token: str) -> dict:
        raise ValueError("invalid token")

    monkeypatch.setattr(middleware.idp_adapter, "get_payload", invalid_payload)

    with pytest.raises(HTTPException) as exc:
        await middleware.validate(make_request(authorization="Bearer bad-token"))

    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_dispatch_allows_public_path() -> None:
    async def app(scope, receive, send) -> None:
        await PlainTextResponse("ok")(scope, receive, send)

    middleware = AuthMiddleware(app)

    async def call_next(request: Request) -> Response:
        return PlainTextResponse("ok")

    response = await middleware.dispatch(make_request("/health"), call_next)

    assert response.body == b"ok"


@pytest.mark.asyncio
async def test_dispatch_handles_http_exception(monkeypatch) -> None:
    middleware = AuthMiddleware(lambda scope, receive, send: None)

    async def call_next(request: Request) -> Response:
        return PlainTextResponse("ok")

    async def invalid(request: Request) -> Request:
        raise HTTPException(status_code=401, detail="unauthorized")

    monkeypatch.setattr(middleware, "validate", invalid)

    response = await middleware.dispatch(make_request(authorization="Bearer bad"), call_next)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_dispatch_handles_unexpected_exception(monkeypatch) -> None:
    middleware = AuthMiddleware(lambda scope, receive, send: None)

    async def call_next(request: Request) -> Response:
        return PlainTextResponse("ok")

    async def invalid(request: Request) -> Request:
        raise RuntimeError("boom")

    monkeypatch.setattr(middleware, "validate", invalid)

    response = await middleware.dispatch(make_request(authorization="Bearer bad"), call_next)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
