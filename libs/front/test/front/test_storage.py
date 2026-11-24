"""Unit tests for chat user persistence.

====================================================================================================
Name:         tests/front/test_storage.py
Description:  Cover user credentials and per-user chat persistence
Project:      agile-monkeys
Date:         2026-11-19
Status:       Development

Copyright ©2026 All rights reserved.
====================================================================================================
"""

from pathlib import Path

from agile_monkeys.front.storage import ChatUserStore


def test_initializes_default_users(tmp_path: Path) -> None:
    store = ChatUserStore(tmp_path)

    assert store.login(" alice ", "1234")
    assert store.login("bob", "1234")
    assert store.login("charlie", "1234")
    assert not store.login("alice", "wrong")
    assert not store.login("unknown", "1234")


def test_existing_users_file_is_not_reseeded(tmp_path: Path) -> None:
    store = ChatUserStore(tmp_path)
    store.users_path.write_text('{"custom": {"password_hash": "test"}}', encoding="utf-8")

    reloaded = ChatUserStore(tmp_path)

    assert reloaded.login("custom", "") is False
    assert reloaded.users_path.read_text(encoding="utf-8") == '{"custom": {"password_hash": "test"}}'


def test_load_chat_returns_empty_state_when_missing(tmp_path: Path) -> None:
    store = ChatUserStore(tmp_path)

    assert store.load_chat("Alice") == {"conversation_id": None, "messages": []}


def test_save_and_load_chat(tmp_path: Path) -> None:
    store = ChatUserStore(tmp_path)
    messages = [{"role": "user", "content": "Hola"}, {"role": "assistant", "content": "Qué tal"}]

    store.save_chat("Alice.User", "conversation-123", messages)

    assert store.load_chat("Alice.User") == {
        "conversation_id": "conversation-123",
        "messages": messages,
    }
    assert store._chat_file("Alice!User") == tmp_path / "chat_aliceuser.json"
