"""Phase 6 - render an investigation as a Markdown incident report.

Log-derived text is sanitized and escaped before it is placed in the report, so
hostile log content cannot inject Markdown, HTML or links into it.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from agent.investigator import Investigation
from guardrails.sanitize import sanitize_text
from rag.retriever import HybridRetriever

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#|!~])")
_TIMELINE_COLUMNS = ("timestamp", "event_type", "host", "process", "src_ip", "dst_ip", "dst_port")


def escape(value: object, max_chars: int = 160) -> str:
    """Make untrusted text safe to embed in Markdown (single line, specials escaped)."""
    text = sanitize_text(value, max_chars).replace("\n", " ").replace("\t", " ")
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def _timeline(events: pd.DataFrame) -> list[str]:
    columns = [column for column in _TIMELINE_COLUMNS if column in events.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in events.sort_values("timestamp")[columns].astype(object).to_dict(orient="records"):
        cells = ["" if pd.isna(row[c]) else escape(row[c], 80) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_report(
    alert: dict[str, Any],
    investigation: Investigation,
    *,
    events: pd.DataFrame | None = None,
    retriever: HybridRetriever | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Build the Markdown report: summary, timeline, ATT&CK mapping, evidence, remediation."""
    verdict = investigation.verdict
    stamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    outcome = "MALICIOUS" if verdict.is_malicious else "NOT MALICIOUS"
    lines = [
        f"# Incident report - {escape(alert.get('event_id', 'unknown alert'), 80)}",
        "",
        f"_Generated {stamp} by SentinelAI. Machine-drafted; verify before acting._",
        "",
        "## Summary",
        "",
        f"- **Verdict:** {outcome}",
        f"- **Severity:** {verdict.severity}",
        f"- **Confidence:** {verdict.confidence:.0%}",
        f"- **Human review required:** {'yes' if verdict.needs_human_review else 'no'}",
        "",
    ]
    if events is not None and not events.empty:
        lines += ["## Timeline", "", *_timeline(events), ""]

    lines += ["## MITRE ATT&CK mapping", ""]
    mitigations: list[str] = []
    if not verdict.attack_techniques:
        lines.append("No techniques mapped.")
    for technique in verdict.attack_techniques:
        document = retriever.get(technique) if retriever is not None else None
        if document is None:
            lines.append(f"- **{technique}**")
            continue
        tactics = ", ".join(document.tactics) or "n/a"
        lines.append(f"- **[{technique}]({document.url})** {document.title} - tactics: {tactics}")
        mitigations += [m for m in document.mitigations if m not in mitigations]

    lines += ["", "## Evidence", ""]
    lines += [f"- {escape(item, 400)}" for item in verdict.evidence] or ["No evidence recorded."]

    lines += ["", "## Remediation", ""]
    for action in verdict.recommended_actions:
        gate = " _(requires human approval)_" if action.requires_human_approval else ""
        lines.append(f"- [ ] {escape(action.action, 400)}{gate}")
    if mitigations:
        lines += ["", "ATT&CK mitigations for the mapped techniques:", ""]
        lines += [f"- {mitigation}" for mitigation in mitigations[:10]]
    if not verdict.recommended_actions and not mitigations:
        lines.append("No actions recommended.")

    if verdict.notes:
        lines += ["", "## Guardrail notes", ""] + [f"- {escape(note, 300)}" for note in verdict.notes]
    lines += ["", "## Investigation trace", "", f"{len(investigation.trace)} steps, "
              f"{investigation.input_tokens} input / {investigation.output_tokens} output tokens.", ""]  # fmt: skip
    for step in investigation.trace:
        if "tool" in step:
            lines.append(f"1. tool `{step['tool']}` {escape(step['arguments'], 200)}")
        else:
            lines.append("1. model produced a verdict")
    return "\n".join(lines) + "\n"
