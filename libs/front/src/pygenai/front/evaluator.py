"""Evaluator module for computing metrics and evidence from multi-turn `/ask` replies."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TurnEvaluation:
    """Evaluation result and supporting evidence for a single conversation turn."""

    turn_index: int
    user_message: str
    assistant_response: str
    conversation_id: str
    duration_ms: int
    tools_used: list[str]
    passed: bool
    reasoning: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ConversationEvaluationReport:
    """Aggregated metrics, transcript, reasoned judgment, and evidence for a run."""

    conversation_id: str
    turn_count: int
    passed_turns: int
    pass_rate: float
    total_duration_ms: int
    avg_duration_ms: float
    tools_invoked: list[str]
    snapshot_summary: dict[str, Any]
    overall_passed: bool
    reasoned_evaluation: str
    transcript: list[dict[str, str]]
    turn_evaluations: list[TurnEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report into a JSON-serializable dictionary."""
        return asdict(self)


class ConversationEvaluator:
    """Evaluates a sequence of user/assistant turns and builds supporting evidence."""

    def evaluate(
        self,
        turns: list[tuple[str, Any]],
        expected_keywords_per_turn: list[list[str]] | None = None,
    ) -> ConversationEvaluationReport:
        """Evaluate a multi-turn conversation and return metrics + evidence.

        Args:
            turns: Ordered list of `(user_message, ask_reply)` pairs in one conversation.
            expected_keywords_per_turn: Optional list of keyword lists per turn.

        Returns:
            A `ConversationEvaluationReport` with transcript, metrics, reasoning, and evidence.
        """
        keywords_list = expected_keywords_per_turn or [[] for _ in turns]
        turn_evaluations: list[TurnEvaluation] = []
        transcript: list[dict[str, str]] = []
        all_tools: list[str] = []
        conversation_id: str = ""
        snapshot_summary: dict[str, Any] = {}

        for idx, (user_msg, reply) in enumerate(turns, start=1):
            conv_id: str = str(getattr(reply, "conversation_id", "") or "")
            if conv_id and not conversation_id:
                conversation_id = conv_id

            duration_ms: int = int(getattr(reply, "duration_ms", 0) or 0)
            raw_text: str = str(getattr(reply, "response", "") or getattr(reply, "answer", "") or "")
            snapshot: dict[str, Any] = dict(getattr(reply, "snapshot", {}) or {})
            payload_data: dict[str, Any] = dict(getattr(reply, "data", {}) or {})

            if snapshot and not snapshot_summary:
                snapshot_summary = {
                    "model": snapshot.get("model"),
                    "promptRevision": snapshot.get("promptRevision") or snapshot.get("revision"),
                    "knowledgeVersionId": snapshot.get("knowledgeVersionId"),
                }

            traces: list[Any] = payload_data.get("traces", []) if isinstance(payload_data.get("traces"), list) else []
            tools_used: list[str] = [str(t.get("tool")) for t in traces if isinstance(t, dict) and t.get("tool")]
            all_tools.extend(tools_used)

            expected_kws: list[str] = keywords_list[idx - 1] if idx - 1 < len(keywords_list) else []
            missing_kws: list[str] = [kw for kw in expected_kws if kw.lower() not in raw_text.lower()]

            reasons: list[str] = []
            passed: bool = True

            if not raw_text.strip():
                passed = False
                reasons.append("Empty response returned by the agent.")
            else:
                reasons.append("Non-empty response returned.")

            if idx > 1 and not conv_id:
                passed = False
                reasons.append("Missing conversation_id on follow-up turn.")
            elif idx > 1:
                reasons.append(f"Context preserved under conversation_id '{conv_id}'.")

            if missing_kws:
                passed = False
                reasons.append(f"Missing expected keywords: {missing_kws}.")
            elif expected_kws:
                reasons.append(f"Matched expected keywords: {expected_kws}.")

            evidence: dict[str, Any] = {
                "conversation_id": conv_id,
                "duration_ms": duration_ms,
                "tools_used": tools_used,
                "tokens": payload_data.get("tokens"),
                "created_at": payload_data.get("createdAt"),
                "snapshot": snapshot,
            }

            turn_evaluations.append(
                TurnEvaluation(
                    turn_index=idx,
                    user_message=user_msg,
                    assistant_response=raw_text,
                    conversation_id=conv_id,
                    duration_ms=duration_ms,
                    tools_used=tools_used,
                    passed=passed,
                    reasoning=" ".join(reasons),
                    evidence=evidence,
                )
            )

            transcript.append({"role": "user", "content": user_msg})
            transcript.append({"role": "assistant", "content": raw_text})

        turn_count: int = len(turn_evaluations)
        passed_turns: int = sum(1 for item in turn_evaluations if item.passed)
        pass_rate: float = round((passed_turns / turn_count) * 100.0, 2) if turn_count else 0.0
        total_duration_ms: int = sum(item.duration_ms for item in turn_evaluations)
        avg_duration_ms: float = round(total_duration_ms / turn_count, 2) if turn_count else 0.0
        unique_tools: list[str] = list(dict.fromkeys(all_tools))
        min_turns_ok: bool = turn_count >= 3
        overall_passed: bool = min_turns_ok and (passed_turns == turn_count) and turn_count > 0

        reasoned_parts: list[str] = [
            f"Evaluated {turn_count} turn(s) in isolated conversation '{conversation_id or 'N/A'}'.",
            f"Passed {passed_turns}/{turn_count} turns ({pass_rate}%).",
            f"Total latency: {total_duration_ms} ms (avg {avg_duration_ms} ms/turn).",
            f"Tools observed in traces: {unique_tools or ['none']}.",
        ]
        if not min_turns_ok:
            reasoned_parts.append("Note: scenario has fewer than 3 turns required for a full multi-turn check.")

        return ConversationEvaluationReport(
            conversation_id=conversation_id,
            turn_count=turn_count,
            passed_turns=passed_turns,
            pass_rate=pass_rate,
            total_duration_ms=total_duration_ms,
            avg_duration_ms=avg_duration_ms,
            tools_invoked=unique_tools,
            snapshot_summary=snapshot_summary,
            overall_passed=overall_passed,
            reasoned_evaluation=" ".join(reasoned_parts),
            transcript=transcript,
            turn_evaluations=turn_evaluations,
        )
