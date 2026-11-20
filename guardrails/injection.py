"""Heuristic prompt-injection detector for log content.

This is a pattern-based detector, not the fine-tuned DistilBERT classifier from
the project plan (not built yet). It is deliberately high-recall: a hit never
blocks an investigation, it only stops the agent from auto-closing the alert.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from guardrails.sanitize import sanitize_text

_PATTERNS: dict[str, re.Pattern[str]] = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b"
        r".{0,40}\b(instruction|prompt|rule|direction)s?\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "addresses_the_model": re.compile(
        r"\b(ai|llm|assistant|analyst|agent|model|chatbot)\b.{0,30}"
        r"\b(must|should|shall|is instructed to|you are now|notice)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "role_marker": re.compile(r"(^|\n)\s*(system|assistant|developer)\s*:|<\|im_start\|>|\[INST\]", re.IGNORECASE),
    "verdict_steering": re.compile(
        r"\b(classify|mark|label|report|treat)\b.{0,60}\b(benign|safe|false positive|harmless|whitelisted)\b"
        r"|\bclose (the|this) (alert|incident|ticket)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "tool_steering": re.compile(
        r"\b(call|invoke|run|execute|use)\b.{0,30}\b(tool|function)\b", re.IGNORECASE | re.DOTALL
    ),
}


@dataclass(frozen=True)
class InjectionFinding:
    """Result of scanning one piece of untrusted text."""

    is_suspicious: bool
    matched: tuple[str, ...]


def detect_injection(text: object) -> InjectionFinding:
    """Flag text that looks like instructions aimed at an AI system."""
    cleaned = sanitize_text(text, 20_000)
    matched = tuple(name for name, pattern in _PATTERNS.items() if pattern.search(cleaned))
    return InjectionFinding(is_suspicious=bool(matched), matched=matched)
