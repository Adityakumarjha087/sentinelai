"""Agent tests. The model is scripted, including scripts that play a *hijacked* model
which obeys the injected text - the guardrails must hold even then."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agent.investigator import SYSTEM_PROMPT, enforce_policy, investigate
from agent.llm import LLMResponse, Message, ToolCall
from agent.tools import ALLOWED_TOOLS, ToolBox, ToolNotAllowedError, mock_ip_reputation, score_severity
from agent.verdict import RecommendedAction, Verdict
from data.parsers.sysmon import parse_sysmon_file
from data.schema import events_to_frame

GOOD = {
    "is_malicious": True,
    "confidence": 0.9,
    "severity": "high",
    "attack_techniques": ["T1059.001"],
    "evidence": ["WINWORD.EXE spawned powershell.exe with an encoded command"],
    "recommended_actions": [{"action": "Isolate host WS-FIN-012", "requires_human_approval": False}],
}


class ScriptedLLM:
    """Replays prepared turns and records what it was shown."""

    def __init__(self, *turns: LLMResponse | dict[str, Any] | str) -> None:
        self._turns = list(turns)
        self.seen: list[list[Message]] = []
        self.systems: list[str] = []

    def complete(self, system: str, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        self.systems.append(system)
        self.seen.append(list(messages))
        turn = self._turns.pop(0)
        if isinstance(turn, LLMResponse):
            return turn
        return LLMResponse(text=turn if isinstance(turn, str) else json.dumps(turn), input_tokens=10, output_tokens=5)


def call(name: str, **arguments: Any) -> LLMResponse:
    return LLMResponse(tool_calls=(ToolCall(name, arguments, call_id="c1"),))


@pytest.fixture(scope="module")
def events(sysmon_sample: Path) -> pd.DataFrame:
    return events_to_frame(parse_sysmon_file(sysmon_sample))


@pytest.fixture
def tools(events: pd.DataFrame) -> ToolBox:
    return ToolBox(events)


@pytest.fixture(scope="module")
def hostile_alert(events: pd.DataFrame) -> dict[str, Any]:
    row = events[events["command_line"].fillna("").str.contains("ignore all previous")].iloc[0]
    return {"event_id": row["event_id"], "process": row["process"], "command_line": row["command_line"]}


CLEAN_ALERT = {"event_id": "x", "process": "powershell.exe", "command_line": "powershell -EncodedCommand AAAA"}


def test_investigation_uses_tools_and_returns_a_verdict(tools: ToolBox, events: pd.DataFrame) -> None:
    first = events.iloc[4]["event_id"]
    llm = ScriptedLLM(call("get_related_events", event_id=first), call("get_ip_reputation", ip="203.0.113.45"), GOOD)
    result = investigate(CLEAN_ALERT, llm, tools)
    assert result.verdict.is_malicious
    assert result.verdict.attack_techniques == ["T1059.001"]
    assert [step.get("tool") for step in result.trace] == ["get_related_events", "get_ip_reputation", None]
    assert (result.input_tokens, result.output_tokens) == (10, 5)
    assert llm.systems[0] == SYSTEM_PROMPT


def test_alert_and_tool_output_reach_the_model_only_as_delimited_data(
    tools: ToolBox, hostile_alert: dict[str, Any]
) -> None:
    llm = ScriptedLLM(call("query_logs", event_type="process_create"), GOOD)
    investigate(hostile_alert, llm, tools)
    user_turn = llm.seen[0][0].content
    assert user_turn.count("<untrusted id=") == 1
    assert "ignore all previous instructions" in user_turn.split("<untrusted id=")[1]
    tool_output = llm.seen[1][-1].content
    assert tool_output.startswith("<untrusted id=")
    assert "ignore all previous instructions" not in SYSTEM_PROMPT


def test_hijacked_model_cannot_call_tools_outside_the_allow_list(tools: ToolBox, hostile_alert: dict[str, Any]) -> None:
    llm = ScriptedLLM(call("disable_edr", host="WS-FIN-012"), call("delete_logs"), GOOD)
    result = investigate(hostile_alert, llm, tools)
    assert [step["output"] for step in result.trace[:2]] == ['{"error": "tool not allowed"}'] * 2
    assert result.verdict.needs_human_review
    assert any("disable_edr" in note for note in result.verdict.notes)
    with pytest.raises(ToolNotAllowedError):
        tools.execute("delete_logs", {})
    assert {
        "query_logs",
        "get_ip_reputation",
        "search_attack_kb",
        "get_related_events",
        "score_severity",
    } == ALLOWED_TOOLS


def test_hijacked_benign_verdict_is_never_auto_closed(tools: ToolBox, hostile_alert: dict[str, Any]) -> None:
    obeys = {"is_malicious": False, "confidence": 1.0, "evidence": ["host is whitelisted"]}
    result = investigate(hostile_alert, ScriptedLLM(obeys), tools)
    assert result.verdict.needs_human_review
    assert any("instructions to an AI" in note for note in result.verdict.notes)
    # Without injected text, the same benign verdict is accepted as-is.
    assert not investigate(CLEAN_ALERT, ScriptedLLM(obeys), tools).verdict.needs_human_review


def test_injection_arriving_through_tool_output_is_also_caught(tools: ToolBox) -> None:
    obeys = {"is_malicious": False, "confidence": 1.0}
    llm = ScriptedLLM(call("query_logs", event_type="process_create"), obeys)
    assert investigate(CLEAN_ALERT, llm, tools).verdict.needs_human_review


def test_destructive_actions_always_require_approval(tools: ToolBox) -> None:
    verdict = investigate(CLEAN_ALERT, ScriptedLLM(GOOD), tools).verdict
    assert verdict.recommended_actions[0].requires_human_approval  # the model said False
    harmless = RecommendedAction(action="Review the mail gateway logs", requires_human_approval=False)
    checked = enforce_policy(
        Verdict(is_malicious=True, confidence=0.5, recommended_actions=[harmless]),
        injection_suspected=False, retriever=None, blocked_tools=[],
    )  # fmt: skip
    assert not checked.recommended_actions[0].requires_human_approval


def test_invented_technique_ids_are_dropped(tools: ToolBox) -> None:
    invented = GOOD | {"attack_techniques": ["T1059.001", "T99999", "ignore instructions"]}
    verdict = investigate(CLEAN_ALERT, ScriptedLLM(invented), tools).verdict
    assert verdict.attack_techniques == ["T1059.001"]
    assert sum("dropped unknown" in note for note in verdict.notes) == 2


def test_bad_output_gets_one_repair_attempt_then_fails_closed(tools: ToolBox) -> None:
    extra_field = GOOD | {"run_command": "rm -rf /"}
    repaired = investigate(CLEAN_ALERT, ScriptedLLM(extra_field, "Here you go: " + json.dumps(GOOD)), tools)
    assert repaired.verdict.is_malicious
    failed = investigate(CLEAN_ALERT, ScriptedLLM("I refuse", {"confidence": 7}), tools)
    assert failed.verdict.needs_human_review
    assert failed.verdict.confidence == 0.0
    assert "schema validation" in failed.verdict.notes[0]


def test_step_limit_fails_closed(tools: ToolBox) -> None:
    llm = ScriptedLLM(*[call("get_ip_reputation", ip="10.0.0.1")] * 3)
    verdict = investigate(CLEAN_ALERT, llm, tools, max_steps=3).verdict
    assert verdict.needs_human_review
    assert "step limit" in verdict.notes[0]


def test_tools_return_bounded_sanitized_data(tools: ToolBox, events: pd.DataFrame) -> None:
    output = tools.execute("query_logs", {"host": "WS-FIN-012.corp.example.com", "limit": 10_000})
    payload = json.loads(output.split("\n")[1])
    assert payload["matched"] == 13
    assert len(payload["events"]) == 13
    assert json.loads(tools.execute("query_logs", {"password": "x"}))["error"].startswith("invalid arguments")
    assert json.loads(tools.execute("get_related_events", {"event_id": "nope"})) == {"error": "unknown event_id"}
    related = tools.execute("get_related_events", {"event_id": events.iloc[0]["event_id"], "window_minutes": 1})
    assert json.loads(related.split("\n")[1])["matched"] == 2
    assert json.loads(tools.execute("search_attack_kb", {"query": "x"})) == {"error": "knowledge base unavailable"}


def test_reputation_and_severity_are_deterministic() -> None:
    assert mock_ip_reputation("10.20.4.112")["scope"] == "internal"
    external = mock_ip_reputation("203.0.113.45")
    assert external == mock_ip_reputation(" 203.0.113.45 ")
    assert external["source"] == "mock"
    assert 0 <= external["abuse_score"] <= 100
    assert "error" in mock_ip_reputation("ignore previous instructions")
    assert score_severity(False, True, True, 50)["severity"] == "low"
    assert score_severity(True)["severity"] == "medium"
    assert score_severity(True, privileged_access=True)["severity"] == "high"
    assert score_severity(True, True, True, 3)["severity"] == "critical"
