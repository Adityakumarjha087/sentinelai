"""Prompt-injection detector for log content: patterns, plus a classifier when one is loaded.

The patterns always run. When a fine-tuned classifier is registered with
:func:`use_classifier` (the API and dashboard do this when ``sentinel-guard
train`` has been run), text it scores at or above its threshold is flagged
too. Either signal is enough. A hit never blocks an investigation; it only
stops the agent from auto-closing the alert.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

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


class InjectionScorer(Protocol):
    """A model that scores text for prompt injection (see :mod:`guardrails.classifier`)."""

    threshold: float

    def score(self, texts: Sequence[str]) -> list[float]: ...


_classifier: InjectionScorer | None = None


def use_classifier(classifier: InjectionScorer | None) -> None:
    """Register (or with ``None``, remove) the classifier :func:`detect_injection` consults."""
    global _classifier
    _classifier = classifier


def active_classifier() -> InjectionScorer | None:
    return _classifier


@dataclass(frozen=True)
class InjectionFinding:
    """Result of scanning one piece of untrusted text."""

    is_suspicious: bool
    matched: tuple[str, ...]
    classifier_score: float | None = None


def pattern_matches(text: str) -> tuple[str, ...]:
    """Names of the patterns that match ``text``."""
    return tuple(name for name, pattern in _PATTERNS.items() if pattern.search(text))


def detect_injection(text: object) -> InjectionFinding:
    """Flag text that looks like instructions aimed at an AI system."""
    cleaned = sanitize_text(text, 20_000)
    matched = pattern_matches(cleaned)
    score = None
    if _classifier is not None:
        score = _classifier.score([cleaned])[0]
        if score >= _classifier.threshold:
            matched = (*matched, "classifier")
    return InjectionFinding(is_suspicious=bool(matched), matched=matched, classifier_score=score)
