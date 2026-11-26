"""Tests for IDP adapters and factory."""

from types import SimpleNamespace

import pytest

from pygenai.core.middlewares.idp.idp_adapter import IDPAdapter
from pygenai.core.middlewares.idp.idp_factory import IDPFactory
from pygenai.core.middlewares.idp.keycloak_adapter import KeycloakAdapter


class ConcreteAdapter(IDPAdapter):
    pass


def test_unverified_and_verified_payload(monkeypatch) -> None:
    monkeypatch.setattr("pygenai.core.middlewares.idp.idp_adapter.PyJWKClient", lambda url: SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key="public-key")))

    unverified = ConcreteAdapter("https://idp/jwks", "audience", "issuer")
    monkeypatch.setattr("pygenai.core.middlewares.idp.idp_adapter.jwt.decode", lambda token, key=None, **kwargs: {"sub": "user", "mode": "unverified", **kwargs})

    payload = unverified.get_payload("token")

    assert payload["sub"] == "user"
    assert payload["mode"] == "unverified"

    verified = ConcreteAdapter("https://idp/jwks", "audience", "issuer", verify_token=True)
    payload = verified.get_payload("token")

    assert payload["sub"] == "user"
    assert payload["audience"] == "audience"


def test_idp_roles_and_key(monkeypatch) -> None:
    monkeypatch.setattr("pygenai.core.middlewares.idp.idp_adapter.PyJWKClient", lambda url: SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key="key")))
    adapter = ConcreteAdapter("https://idp/jwks", "audience", "issuer")

    assert adapter.get_roles({"roles": ["admin", "admin"]}) == {"admin"}
    assert adapter._get_key("token") == "key"


def test_keycloak_urls_roles_and_verified_payload(monkeypatch) -> None:
    monkeypatch.setattr("pygenai.core.middlewares.idp.idp_adapter.PyJWKClient", lambda url: SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key="key")))
    monkeypatch.setattr("pygenai.core.middlewares.idp.keycloak_adapter.jwt.decode", lambda token, key, **kwargs: {"sub": "keycloak-user", **kwargs})

    adapter = KeycloakAdapter("https://idp", "client", "realm", verify_token=True)

    assert adapter.jwks_url == "https://idp/realms/realm/protocol/openid-connect/certs"
    assert adapter.issuer == "https://idp/realms/realm"
    assert adapter.get_roles({"resource_access": {"client": {"roles": ["reader", "writer"]}}}) == {"reader", "writer"}
    assert adapter.get_payload("token")["sub"] == "keycloak-user"


def test_keycloak_falls_back_to_default_verification(monkeypatch) -> None:
    monkeypatch.setattr("pygenai.core.middlewares.idp.idp_adapter.PyJWKClient", lambda url: SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key="key")))

    def fail_decode(*args: object, **kwargs: object) -> dict:
        raise ValueError("account audience failed")

    monkeypatch.setattr("pygenai.core.middlewares.idp.keycloak_adapter.jwt.decode", fail_decode)
    monkeypatch.setattr("pygenai.core.middlewares.idp.idp_adapter.jwt.decode", lambda token, key=None, **kwargs: {"sub": "fallback"})

    adapter = KeycloakAdapter("https://idp", "client", "realm", verify_token=True)

    assert adapter.get_payload("token") == {"sub": "fallback"}


def test_factory_builds_keycloak(monkeypatch) -> None:
    monkeypatch.setenv("IDP_URL", "https://idp")
    monkeypatch.setenv("IDP_AUDIENCE", "client")
    monkeypatch.setenv("IDP_ISSUER", "realm")

    adapter = IDPFactory("KEYCLOAK", verify_token=True).get_idp()

    assert isinstance(adapter, KeycloakAdapter)
    assert adapter.verify_token is True


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown IDP provider"):
        IDPFactory("unknown").get_idp()
