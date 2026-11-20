"""Offline stand-ins for an LLM, so the harness runs without an API key.

These are *not* language models. ``RuleBasedAnalyst`` is a keyword baseline that
any real model should beat; ``GullibleAnalyst`` additionally obeys injected text,
which measures how much the guardrails alone protect a fully hijacked model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.llm import LLMResponse, Message

_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"encodedcommand|winword", re.I), "T1059.001", "PowerShell launched with an encoded command"),
    (re.compile(r"lsass", re.I), "T1003.001", "a process opened lsass.exe memory"),
    (re.compile(r"currentversion\\+run", re.I), "T1547.001", "a Run key was set for persistence"),
    (re.compile(r"schtasks.*/create", re.I), "T1053.005", "a scheduled task was created"),
    # Flow alerts state the port count for every flow, so only a high count is a scan.
    (re.compile(r"\b(?:[2-9]\d|\d{3,}) distinct destination ports", re.I), "T1046", "one source contacted many ports"),
    (re.compile(r"failed logins", re.I), "T1110", "repeated failed logins from one source"),
    (re.compile(r"requests per minute", re.I), "T1498", "request flood against one server"),
]
_OBEY = re.compile(r"benign|whitelisted|harmless|false positive|not malicious|is_malicious=false|safe", re.I)


def _alert_text(messages: list[Message]) -> str:
    return messages[0].content.split("\n", 2)[-1]


class RuleBasedAnalyst:
    """Keyword rules over the alert text. Deterministic, free, and limited."""

    obeys_injection = False

    def complete(self, system: str, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        text = _alert_text(messages)
        hits = [(technique, evidence) for pattern, technique, evidence in _RULES if pattern.search(text)]
        malicious = bool(hits)
        if self.obeys_injection and _OBEY.search(text):
            malicious, hits = False, []
        verdict = {
            "is_malicious": malicious,
            "confidence": 0.8 if malicious else 0.6,
            "severity": "high" if malicious else "low",
            "attack_techniques": [technique for technique, _ in hits],
            "evidence": [evidence for _, evidence in hits],
            "recommended_actions": [{"action": "Isolate the host pending review"}] if malicious else [],
        }
        return LLMResponse(text=json.dumps(verdict))


class GullibleAnalyst(RuleBasedAnalyst):
    """The same rules, but it does whatever injected text tells it to."""

    obeys_injection = True
