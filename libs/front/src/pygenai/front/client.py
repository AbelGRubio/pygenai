"""HTTP client for sending chat turns to the Hiring Lab Private API `/ask` endpoint."""

import ast
import json
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from pygenai.front.settings import get_settings


def parse_dict_string(value: Any) -> dict[str, Any]:
    """Convierte un string con formato de diccionario (JSON o literal de Python) en un dict."""
    if isinstance(value, dict):
        return value

    if not isinstance(value, str) or not value.strip():
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}

    return parsed if isinstance(parsed, dict) else {}


class AskAPIError(RuntimeError):
    """Raised when `POST /ask` returns a non-successful HTTP response."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Initialize the API error with status and response detail."""
        super().__init__(f"API error [{status_code}]: {detail}")
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class AskReply:
    """Response payload data returned by a `/ask` turn."""

    answer: str
    conversation_id: str
    duration_ms: int
    response: str
    snapshot: dict[str, Any]
    data: dict[str, Any]


class AskClient:
    """Minimal client for continuing multi-turn conversations via `/ask`."""

    def __init__(
        self,
        base_url: str | None = None,
        project_key: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        """Initialize the client from explicit arguments or environment variables.

        Args:
            base_url: Project API base URL (defaults to `HIRING_LAB_BASE_URL`).
            project_key: Secret project API key (defaults to `HIRING_LAB_PROJECT_KEY`).
            timeout_seconds: Request timeout in seconds.
        """
        self.base_url: str = base_url or ""
        self.project_key: str = project_key or ""
        if not self.base_url or not self.project_key:
            config = get_settings()
            self.base_url = self.base_url or config.hiring_lab_base_url
            self.project_key = self.project_key or config.hiring_lab_project_key
        self.timeout_seconds: int = timeout_seconds

        if not self.base_url or not self.project_key:
            raise ValueError("Missing HIRING_LAB_BASE_URL or HIRING_LAB_PROJECT_KEY in environment.")

    def send_message(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> AskReply:
        """Send a single user turn to `POST /ask`.

        Args:
            question: User text message.
            conversation_id: Existing conversation UUID to keep context, or None to start one.

        Returns:
            An `AskReply` with the assistant answer and active `conversation_id`.
        """
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.project_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": str(uuid.uuid4()),
        }
        payload: dict[str, str] = {"question": question}
        if conversation_id:
            payload["conversationId"] = conversation_id

        response = requests.post(
            f"{self.base_url}/ask",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )

        if not response.ok:
            raise AskAPIError(response.status_code, response.text)

        data: dict[str, object] = response.json()
        answer = parse_dict_string(data.get("answer", ""))
        response = answer.get("text_response", "")
        duration_value = data.get("durationMs", 0)
        duration_ms = duration_value if isinstance(duration_value, int) else 0
        snapshot_value = data.get("snapshot", {})
        snapshot = snapshot_value if isinstance(snapshot_value, dict) else {}
        returned_conversation_id = data.get("conversationId")
        if not isinstance(returned_conversation_id, str) or not returned_conversation_id:
            returned_conversation_id = conversation_id or ""
        return AskReply(
            answer=str(answer),
            conversation_id=returned_conversation_id,
            duration_ms=duration_ms,
            response=str(response),
            snapshot=snapshot,
            data=data,
        )
