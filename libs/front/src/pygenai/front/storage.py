"""Simple user login and per-user chat context persistence."""

import hashlib
import json
from pathlib import Path
from typing import Any


class ChatUserStore:
    """Stores users and preserves each user's conversation history and ID."""

    def __init__(self, data_dir: str | Path = ".chat_data") -> None:
        """Initialize storage folder and seed default demo users if absent.

        Args:
            data_dir: Directory where user credentials and chat histories are stored.
        """
        self.data_dir: Path = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.users_path: Path = self.data_dir / "users.json"
        self._init_users()

    @staticmethod
    def _hash(password: str) -> str:
        """Return SHA-256 hex digest for a password string."""
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _init_users(self) -> None:
        """Create default users on first run."""
        if not self.users_path.exists():
            users: dict[str, dict[str, str]] = {
                "alice": {"password_hash": self._hash("1234")},
                "bob": {"password_hash": self._hash("1234")},
                "charlie": {"password_hash": self._hash("1234")},
            }
            self.users_path.write_text(json.dumps(users, indent=2), encoding="utf-8")

    def login(self, username: str, password: str) -> bool:
        """Validate username and password credentials.

        Args:
            username: Submitted username.
            password: Submitted plaintext password.

        Returns:
            True if credentials match a stored user, False otherwise.
        """
        clean_user: str = username.strip().lower()
        users: dict[str, Any] = json.loads(self.users_path.read_text(encoding="utf-8"))
        user_record = users.get(clean_user)
        if not user_record:
            return False
        return user_record.get("password_hash") == self._hash(password)

    def _chat_file(self, username: str) -> Path:
        """Return the chat state file path for a specific user."""
        safe_user: str = "".join(ch for ch in username.lower() if ch.isalnum() or ch in {"_", "-"})
        return self.data_dir / f"chat_{safe_user}.json"

    def load_chat(self, username: str) -> dict[str, Any]:
        """Load saved `conversation_id` and `messages` for a user.

        Args:
            username: Authenticated username.

        Returns:
            Dictionary containing `conversation_id` and `messages`.
        """
        path: Path = self._chat_file(username)
        if not path.exists():
            return {"conversation_id": None, "messages": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_chat(
        self,
        username: str,
        conversation_id: str | None,
        messages: list[dict[str, str]],
    ) -> None:
        """Persist `conversation_id` and `messages` for a user.

        Args:
            username: Authenticated username.
            conversation_id: Current conversation UUID from `/ask`.
            messages: Ordered list of `{"role": ..., "content": ...}` messages.
        """
        payload: dict[str, Any] = {
            "conversation_id": conversation_id,
            "messages": messages,
        }
        self._chat_file(username).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
