"""Streamlit app with login, scrollable chat tab, and evaluation/evidence tab.

Interactive Streamlit front end for the Agile Monkeys Hiring Lab.

The front end provides a browser-based interface for two connected workflows:
conversing with a Hiring Lab agent and inspecting the resulting evidence.
After authentication, users can send messages, keep a persistent conversation
identifier, start a new chat, and review turn-level evaluation metrics.

## Authentication

The login screen authenticates demo users against the local JSON user store.
Each authenticated user receives an isolated chat history and is presented with
`alice`, `bob`, or `charlie` demo credentials with password `1234`.

![Login screen](../../../imgs/login.png)

## Chat

The Chat tab displays the current conversation in a scrollable message area.
The first message creates an API conversation identifier; subsequent messages
reuse that identifier to preserve context. **New chat** clears the local history
and starts a new identifier.

The API-backed chat requires `HIRING_LAB_BASE_URL` and
`HIRING_LAB_PROJECT_KEY` environment variables. User and conversation state are
stored locally in `.chat_data/` as JSON files; credentials and real secrets
should not be added to this demo storage.

![Chat conversation](../../../imgs/chat.png)

## Evaluation and evidence

The Evaluation & Evidence tab reconstructs the full transcript and summarizes
agent behavior. It reports turn count, pass rate, total latency, and distinct
tools used. It also includes reasoned evaluation text and expandable evidence
for every turn, such as the conversation identifier, duration, tool traces,
tokens, creation time, and API snapshot.

![Evaluation metrics and transcript evidence](../../../imgs/eval.png)

## Launch

The front end is exposed through the project CLI:

```console
agile-monkeys front
```

Use `--host`, `--port`, and `--no-browser` when a custom server address or
headless startup is required.
"""


from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
import streamlit as st

from pygenai.front.client import AskAPIError, AskClient, AskReply
from pygenai.front.evaluator import ConversationEvaluator, ConversationEvaluationReport
from pygenai.front.storage import ChatUserStore

load_dotenv()

st.set_page_config(page_title="Agent Chat & Evaluator", page_icon="💬", layout="centered")


def _build_replies_from_messages(
    messages: list[dict[str, Any]],
    fallback_conversation_id: str,
) -> list[tuple[str, Any]]:
    """Reconstruct `(user_message, reply_like_object)` pairs from stored chat messages.

    Args:
        messages: Ordered chat history list for the active conversation.
        fallback_conversation_id: Active `conversation_id` for the conversation.

    Returns:
        List of `(question, reply_obj)` pairs ready for `ConversationEvaluator.evaluate`.
    """
    pairs: list[tuple[str, Any]] = []
    pending_user: str | None = None

    for msg in messages:
        role: str = str(msg.get("role", ""))
        content: str = str(msg.get("content", ""))

        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            meta: dict[str, Any] = dict(msg.get("meta") or {})
            reply_obj = SimpleNamespace(
                answer=content,
                response=content,
                conversation_id=str(meta.get("conversation_id") or fallback_conversation_id),
                duration_ms=int(meta.get("duration_ms", 0) or 0),
                snapshot=dict(meta.get("snapshot") or {}),
                data=dict(meta.get("data") or {}),
            )
            pairs.append((pending_user, reply_obj))
            pending_user = None

    return pairs


def render_login(store: ChatUserStore) -> None:
    """Render login form before showing the application tabs.

    Args:
        store: User store used to verify credentials.
    """
    st.title("🔐 Login")
    st.caption("Demo users: `alice` / `1234` or `bob` / `1234`")

    with st.form("login"):
        username: str = st.text_input("Username")
        password: str = st.text_input("Password", type="password")
        submit: bool = st.form_submit_button("Enter", use_container_width=True)

        if submit:
            if store.login(username, password):
                st.session_state["user"] = username.strip().lower()
                st.rerun()
            else:
                st.error("Invalid username or password.")


def render_chat_tab(username: str, store: ChatUserStore, client: AskClient | None) -> None:
    """Render the chat tab with internal message scrolling and bottom input.

    Args:
        username: Authenticated username.
        store: Persistence store for the user's chat session.
        client: Initialized API client, or None if env vars are missing.
    """
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            overflow: hidden;
        }
        .stChatInput {
            position: sticky;
            bottom: 0;
            z-index: 99;
            background-color: var(--background-color);
            padding-top: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    chat_data: dict[str, Any] = store.load_chat(username)
    conversation_id: str | None = chat_data.get("conversation_id")
    messages: list[dict[str, Any]] = list(chat_data.get("messages", []))

    top_left, top_right = st.columns([3, 1])
    top_left.caption(
        f"User: `{username}` | Conversation ID: `{conversation_id or 'New on first message'}`"
    )
    if top_right.button("New chat"):
        store.save_chat(username, conversation_id=None, messages=[])
        st.rerun()

    messages_container = st.container(height=450, border=True)
    with messages_container:
        for msg in messages:
            with st.chat_message(str(msg.get("role", "assistant"))):
                st.markdown(str(msg.get("content", "")))

    prompt: str | None = st.chat_input("Write something...")

    if prompt:
        if client is None:
            st.error("Set HIRING_LAB_BASE_URL and HIRING_LAB_PROJECT_KEY first.")
            return

        messages.append({"role": "user", "content": prompt})
        with messages_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"), st.spinner("Thinking..."):
                try:
                    reply: AskReply = client.send_message(
                        question=prompt, conversation_id=conversation_id
                    )
                    st.markdown(reply.response)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": reply.response,
                            "meta": {
                                "conversation_id": reply.conversation_id,
                                "duration_ms": reply.duration_ms,
                                "snapshot": reply.snapshot,
                                "data": reply.data,
                            },
                        }
                    )
                    store.save_chat(
                        username=username,
                        conversation_id=reply.conversation_id,
                        messages=messages,
                    )
                    st.rerun()
                except (AskAPIError, RuntimeError) as exc:
                    st.error(str(exc))

    st.iframe(
        """
        <script>
        const parentDoc = window.parent.document;
        const boxes = parentDoc.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
        if (boxes.length > 0) {
            const box = boxes[boxes.length - 1];
            box.scrollTop = box.scrollHeight;
        }
        </script>
        """,
        height=0,
    )


def render_evaluation_tab(username: str, store: ChatUserStore) -> None:
    """Render full transcript, reasoned evaluation, metrics, and supporting evidence.

    Args:
        username: Authenticated username.
        store: Persistence store containing the user's conversation history.
    """
    chat_data: dict[str, Any] = store.load_chat(username)
    conversation_id: str = str(chat_data.get("conversation_id") or "")
    messages: list[dict[str, Any]] = list(chat_data.get("messages", []))

    turns = _build_replies_from_messages(messages, fallback_conversation_id=conversation_id)
    if not turns:
        st.info("Send at least one message in the Chat tab to inspect transcript and evidence.")
        return

    evaluator = ConversationEvaluator()
    report: ConversationEvaluationReport = evaluator.evaluate(turns=turns)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Turns", report.turn_count)
    m2.metric("Pass Rate", f"{report.pass_rate}%")
    m3.metric("Total Latency", f"{report.total_duration_ms} ms")
    m4.metric("Tools Used", len(report.tools_invoked))

    st.markdown("### Reasoned Evaluation")
    st.write(report.reasoned_evaluation)

    if report.snapshot_summary:
        with st.expander("Snapshot & Project Evidence"):
            st.json(
                {
                    "conversation_id": report.conversation_id,
                    "snapshot_summary": report.snapshot_summary,
                    "tools_invoked": report.tools_invoked,
                }
            )

    st.markdown("### Full Transcript & Turn Evidence")
    for turn_eval in report.turn_evaluations:
        status_icon: str = "✅" if turn_eval.passed else "❌"
        with st.expander(
            f"{status_icon} Turn {turn_eval.turn_index} ({turn_eval.duration_ms} ms)",
            expanded=True,
        ):
            st.markdown(f"**User:** {turn_eval.user_message}")
            st.markdown(f"**Assistant:** {turn_eval.assistant_response}")
            st.markdown(f"**Reasoning:** {turn_eval.reasoning}")
            st.json(turn_eval.evidence)


def main() -> None:
    """Run login gate, chat tab, and evaluation/evidence tab."""
    store = ChatUserStore()

    if "user" not in st.session_state:
        render_login(store)
        return

    username: str = st.session_state["user"]

    with st.sidebar:
        st.write(f"Logged in as **{username}**")
        if st.button("Logout", use_container_width=True):
            st.session_state.pop("user", None)
            st.rerun()

    client: AskClient | None = None
    try:
        client = AskClient()
    except ValueError as exc:
        st.warning(str(exc))

    st.title("💬 Agent Chat & Evaluator")
    chat_tab, eval_tab = st.tabs(["Chat", "Evaluation & Evidence"])

    with chat_tab:
        render_chat_tab(username=username, store=store, client=client)

    with eval_tab:
        render_evaluation_tab(username=username, store=store)


if __name__ == "__main__":
    main()
