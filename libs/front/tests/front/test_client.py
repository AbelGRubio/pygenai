"""Unit tests for the Hiring Lab HTTP chat client.

====================================================================================================
Name:         tests/front/test_client.py
Description:  Cover response parsing, HTTP calls, and API errors
Date:         2026-11-19
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from pygenai.front.client import AskAPIError, AskClient, parse_dict_string


def test_parse_dict_string_accepts_supported_formats() -> None:
    """Verify that supported dictionary representations are parsed correctly.

    The test covers an existing dictionary, JSON and Python-style strings,
    unsupported sequences, invalid text, non-string values, and whitespace.
    """
    value = {"text_response": "already parsed"}

    assert parse_dict_string(value) is value
    assert parse_dict_string('{"text_response": "json"}') == {"text_response": "json"}
    assert parse_dict_string("{'text_response': 'python'}") == {"text_response": "python"}
    assert parse_dict_string("[1, 2]") == {}
    assert parse_dict_string("not: valid") == {}
    assert parse_dict_string(42) == {}
    assert parse_dict_string("   ") == {}


def test_init_requires_configured_base_url_and_project_key() -> None:
    """Verify that client initialization fails when the base URL is missing."""
    with (
        patch(
            "agile_monkeys.front.client.get_settings",
            return_value=SimpleNamespace(hiring_lab_base_url="", hiring_lab_project_key="key"),
        ),
        pytest.raises(ValueError, match="Missing HIRING_LAB_BASE_URL"),
    ):
        AskClient()


def test_init_uses_application_settings() -> None:
    """Verify that client configuration is loaded from application settings."""
    settings = SimpleNamespace(
        hiring_lab_base_url="https://api.example.test",
        hiring_lab_project_key="secret",
    )

    with patch("agile_monkeys.front.client.get_settings", return_value=settings):
        client = AskClient(timeout_seconds=12)

    assert client.base_url == "https://api.example.test"
    assert client.project_key == "secret"
    assert client.timeout_seconds == 12


def test_send_message_posts_payload_and_parses_reply() -> None:
    """Verify the HTTP request payload, headers, timeout, and parsed response."""
    settings = SimpleNamespace(
        hiring_lab_base_url="https://api.example.test",
        hiring_lab_project_key="secret",
    )
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {
        "answer": '{"text_response": "Hello!"}',
        "conversationId": "conversation-123",
        "durationMs": 321,
        "snapshot": {"step": 2},
    }

    with (
        patch("agile_monkeys.front.client.get_settings", return_value=settings),
        patch("agile_monkeys.front.client.requests.post", return_value=response) as post,
        patch("agile_monkeys.front.client.uuid.uuid4", return_value="idem-key"),
    ):
        reply = AskClient(timeout_seconds=15).send_message("Hi", "conversation-123")

    post.assert_called_once()
    args, kwargs = post.call_args
    assert args == ("https://api.example.test/ask",)
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["headers"]["Idempotency-Key"] == "idem-key"
    assert kwargs["json"] == {"question": "Hi", "conversationId": "conversation-123"}
    assert kwargs["timeout"] == 15
    assert reply.answer == "{'text_response': 'Hello!'}"
    assert reply.response == "Hello!"
    assert reply.conversation_id == "conversation-123"
    assert reply.duration_ms == 321
    assert reply.snapshot == {"step": 2}


def test_send_message_handles_defaults_and_invalid_optional_fields() -> None:
    """Verify safe defaults for missing or invalid optional response fields."""
    settings = SimpleNamespace(
        hiring_lab_base_url="https://api.example.test",
        hiring_lab_project_key="secret",
    )
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {
        "answer": 123,
        "conversationId": None,
        "durationMs": "unknown",
        "snapshot": "invalid",
    }

    with (
        patch("agile_monkeys.front.client.get_settings", return_value=settings),
        patch("agile_monkeys.front.client.requests.post", return_value=response),
    ):
        reply = AskClient().send_message("Start")

    assert reply.answer == "{}"
    assert reply.response == ""
    assert reply.conversation_id == ""
    assert reply.duration_ms == 0
    assert reply.snapshot == {}


def test_send_message_raises_api_error() -> None:
    """Verify that unsuccessful HTTP responses raise an API error with status and detail."""
    settings = SimpleNamespace(
        hiring_lab_base_url="https://api.example.test",
        hiring_lab_project_key="secret",
    )
    response = Mock(ok=False, status_code=429, text="Too many requests")

    with (
        patch("agile_monkeys.front.client.get_settings", return_value=settings),
        patch("agile_monkeys.front.client.requests.post", return_value=response),
    ):
        client = AskClient()
        with pytest.raises(AskAPIError) as exc_info:
            client.send_message("Hi")

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail == "Too many requests"
    assert str(exc_info.value) == "API error [429]: Too many requests"
