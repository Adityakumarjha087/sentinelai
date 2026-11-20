"""Phase 6 - render an investigation as an incident report, in Markdown or PDF.

:func:`report_content` decides what a report says, once, as sanitized plain
text; the two renderers only lay it out, so they cannot drift apart.

Log-derived text - and anything a model or API client supplied - is untrusted:

* Markdown: every untrusted field is escaped, so it cannot inject headings,
  HTML, links or code spans.
* PDF: text is written literally (fpdf2 interprets no markup), links are added
  only for knowledge-base URLs on known hosts, and characters the bundled font
  cannot draw are replaced with a visible placeholder instead of vanishing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fontTools.ttLib import TTFont

from agent.investigator import Investigation
from agent.verdict import Verdict
from guardrails.sanitize import sanitize_text
from rag.retriever import HybridRetriever

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#|!~])")
_TIMELINE_COLUMNS = ("timestamp", "event_type", "host", "process", "src_ip", "dst_ip", "dst_port")
_TIMELINE_WEIGHTS = {"timestamp": 25, "event_type": 24, "host": 33, "process": 50, "src_ip": 19, "dst_ip": 19,
                     "dst_port": 12}  # fmt: skip
_LINK_PREFIXES = ("https://attack.mitre.org/", "https://nvd.nist.gov/")
FONT_DIR = Path(__file__).resolve().parent / "fonts"
_FONT = "DejaVu"
_REPLACEMENT = "\N{REPLACEMENT CHARACTER}"


def _plain(value: object, max_chars: int) -> str:
    """Sanitized, truncated, single-line text: the form every renderer starts from."""
    return sanitize_text(value, max_chars).replace("\n", " ").replace("\t", " ")


def _md(text: str) -> str:
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def escape(value: object, max_chars: int = 160) -> str:
    """Make untrusted text safe to embed in Markdown (single line, specials escaped)."""
    return _md(_plain(value, max_chars))


@dataclass(frozen=True)
class TechniqueEntry:
    """One mapped ATT&CK technique, with knowledge-base details when they exist."""

    technique: str
    title: str | None = None
    url: str | None = None
    tactics: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceStep:
    """One investigation step: a tool call (with its arguments) or the final model answer."""

    tool: str | None
    arguments: str = ""


@dataclass(frozen=True)
class ReportContent:
    """Everything a report says, as sanitized plain text."""

    event_id: str
    generated: str
    verdict: Verdict
    timeline_columns: tuple[str, ...]
    timeline_rows: tuple[tuple[str, ...], ...]
    techniques: tuple[TechniqueEntry, ...]
    evidence: tuple[str, ...]
    actions: tuple[tuple[str, bool], ...]
    mitigations: tuple[str, ...]
    notes: tuple[str, ...]
    steps: tuple[TraceStep, ...]
    input_tokens: int
    output_tokens: int

    @property
    def outcome(self) -> str:
        return "MALICIOUS" if self.verdict.is_malicious else "NOT MALICIOUS"


def report_content(
    alert: dict[str, Any],
    investigation: Investigation,
    *,
    events: pd.DataFrame | None = None,
    retriever: HybridRetriever | None = None,
    generated_at: datetime | None = None,
) -> ReportContent:
    """Decide what the report says: summary, timeline, ATT&CK mapping, evidence, remediation, trace."""
    verdict = investigation.verdict
    columns: tuple[str, ...] = ()
    rows: list[tuple[str, ...]] = []
    if events is not None and not events.empty:
        columns = tuple(column for column in _TIMELINE_COLUMNS if column in events.columns)
        for row in events.sort_values("timestamp")[list(columns)].astype(object).to_dict(orient="records"):
            rows.append(tuple("" if pd.isna(row[c]) else _plain(row[c], 80) for c in columns))

    techniques: list[TechniqueEntry] = []
    mitigations: list[str] = []
    for technique in verdict.attack_techniques:
        document = retriever.get(technique) if retriever is not None else None
        if document is None:
            techniques.append(TechniqueEntry(_plain(technique, 40)))
            continue
        techniques.append(TechniqueEntry(document.doc_id, document.title, document.url, tuple(document.tactics)))
        mitigations += [m for m in document.mitigations if m not in mitigations]

    steps = tuple(
        TraceStep(_plain(step["tool"], 60), _plain(step.get("arguments", ""), 200))
        if "tool" in step
        else TraceStep(None)
        for step in investigation.trace
    )
    return ReportContent(
        event_id=_plain(alert.get("event_id", "unknown alert"), 80),
        generated=(generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC"),
        verdict=verdict,
        timeline_columns=columns,
        timeline_rows=tuple(rows),
        techniques=tuple(techniques),
        evidence=tuple(_plain(item, 400) for item in verdict.evidence),
        actions=tuple((_plain(a.action, 400), a.requires_human_approval) for a in verdict.recommended_actions),
        mitigations=tuple(mitigations[:10]),
        notes=tuple(_plain(note, 300) for note in verdict.notes),
        steps=steps,
        input_tokens=investigation.input_tokens,
        output_tokens=investigation.output_tokens,
    )


def render_markdown(content: ReportContent) -> str:
    """Lay the report out as Markdown, escaping every untrusted field."""
    verdict = content.verdict
    lines = [
        f"# Incident report - {_md(content.event_id)}",
        "",
        f"_Generated {content.generated} by SentinelAI. Machine-drafted; verify before acting._",
        "",
        "## Summary",
        "",
        f"- **Verdict:** {content.outcome}",
        f"- **Severity:** {verdict.severity}",
        f"- **Confidence:** {verdict.confidence:.0%}",
        f"- **Human review required:** {'yes' if verdict.needs_human_review else 'no'}",
        "",
    ]
    if content.timeline_rows:
        lines += ["## Timeline", "", "| " + " | ".join(content.timeline_columns) + " |"]
        lines.append("|" + "---|" * len(content.timeline_columns))
        lines += ["| " + " | ".join(_md(cell) for cell in row) + " |" for row in content.timeline_rows]
        lines.append("")

    lines += ["## MITRE ATT&CK mapping", ""]
    if not content.techniques:
        lines.append("No techniques mapped.")
    for entry in content.techniques:
        if entry.title is None:
            lines.append(f"- **{_md(entry.technique)}**")
        else:
            tactics = ", ".join(entry.tactics) or "n/a"
            lines.append(f"- **[{_md(entry.technique)}]({entry.url})** {entry.title} - tactics: {tactics}")

    lines += ["", "## Evidence", ""]
    lines += [f"- {_md(item)}" for item in content.evidence] or ["No evidence recorded."]

    lines += ["", "## Remediation", ""]
    for action, gated in content.actions:
        lines.append(f"- [ ] {_md(action)}{' _(requires human approval)_' if gated else ''}")
    if content.mitigations:
        lines += ["", "ATT&CK mitigations for the mapped techniques:", ""]
        lines += [f"- {mitigation}" for mitigation in content.mitigations]
    if not content.actions and not content.mitigations:
        lines.append("No actions recommended.")

    if content.notes:
        lines += ["", "## Guardrail notes", ""] + [f"- {_md(note)}" for note in content.notes]
    lines += ["", "## Investigation trace", "", f"{len(content.steps)} steps, "
              f"{content.input_tokens} input / {content.output_tokens} output tokens.", ""]  # fmt: skip
    for step in content.steps:
        if step.tool is None:
            lines.append("1. model produced a verdict")
        else:
            tool = step.tool.replace("`", "'")  # a code span cannot be escaped from inside
            lines.append(f"1. tool `{tool}` {_md(step.arguments)}")
    return "\n".join(lines) + "\n"


def render_report(
    alert: dict[str, Any],
    investigation: Investigation,
    *,
    events: pd.DataFrame | None = None,
    retriever: HybridRetriever | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Build the Markdown report: summary, timeline, ATT&CK mapping, evidence, remediation."""
    content = report_content(alert, investigation, events=events, retriever=retriever, generated_at=generated_at)
    return render_markdown(content)


@lru_cache(maxsize=1)
def _drawable() -> frozenset[int]:
    """Code points both bundled font weights can draw."""
    regular, bold = (set(TTFont(FONT_DIR / name).getBestCmap()) for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"))
    return frozenset(regular & bold)


def printable(text: str) -> str:
    """Replace characters the PDF font cannot draw, so they show up as missing instead of vanishing."""
    drawable = _drawable()
    return "".join(ch if ord(ch) in drawable or ch == "\n" else _REPLACEMENT for ch in text)


def render_pdf(content: ReportContent) -> bytes:
    """Lay the report out as an A4 PDF. Text is written literally; no markup is interpreted."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    class ReportPDF(FPDF):
        def footer(self) -> None:
            self.set_y(-12)
            self.set_font(_FONT, size=7)
            self.set_text_color(110, 110, 110)
            label = printable(f"SentinelAI incident report {content.event_id[:40]}")
            self.cell(0, 5, f"{label} - machine-drafted; verify before acting - page {self.page_no()}/{{nb}}")

    pdf = ReportPDF(format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_font(_FONT, "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font(_FONT, "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.set_title(printable(f"Incident report {content.event_id}"))
    pdf.set_creator("SentinelAI")
    pdf.set_creation_date(datetime.strptime(content.generated, "%Y-%m-%d %H:%M UTC").replace(tzinfo=UTC))
    pdf.add_page()

    def text(body: str, *, size: float = 9.5, bold: bool = False, gray: bool = False, indent: float = 0) -> None:
        pdf.set_font(_FONT, "B" if bold else "", size)
        pdf.set_text_color(*((110, 110, 110) if gray else (20, 20, 20)))
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(0, size * 0.5, printable(body), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def heading(title: str) -> None:
        pdf.ln(3)
        text(title, size=12, bold=True)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.l_margin, pdf.get_y() + 0.5, pdf.w - pdf.r_margin, pdf.get_y() + 0.5)
        pdf.ln(2)

    def bullets(items: list[str], *, empty: str, marker: str = "\N{BULLET}") -> None:
        for item in items:
            text(f"{marker} {item}", indent=2)
        if not items and empty:
            text(empty, gray=True)

    verdict = content.verdict
    text("Incident report", size=17, bold=True)
    text(content.event_id, size=11)
    text(f"Generated {content.generated} by SentinelAI. Machine-drafted; verify before acting.", size=8.5, gray=True)

    heading("Summary")
    pdf.set_font(_FONT, size=9.5)
    with pdf.table(col_widths=(45, 135), first_row_as_headings=False, line_height=6, text_align="LEFT") as table:
        for label, value in (
            ("Verdict", content.outcome),
            ("Severity", verdict.severity),
            ("Confidence", f"{verdict.confidence:.0%}"),
            ("Human review required", "yes" if verdict.needs_human_review else "no"),
        ):
            row = table.row()
            row.cell(label)
            row.cell(value)

    if content.timeline_rows:
        heading("Timeline")
        pdf.set_font(_FONT, size=7)
        columns = content.timeline_columns
        weights = [_TIMELINE_WEIGHTS.get(column, 20) for column in columns]
        stamp = columns.index("timestamp") if "timestamp" in columns else -1
        header = tuple(f"{column} (UTC)" if index == stamp else column for index, column in enumerate(columns))
        with pdf.table(col_widths=weights, line_height=4, text_align="LEFT") as table:
            for number, cells in enumerate((header, *content.timeline_rows)):
                row = table.row()
                for index, cell in enumerate(cells):
                    # Whole seconds are enough on paper; the Markdown report keeps full precision.
                    row.cell(printable(cell[:19] if index == stamp and number else cell))

    heading("MITRE ATT&CK mapping")
    if not content.techniques:
        text("No techniques mapped.", gray=True)
    for entry in content.techniques:
        pdf.set_x(pdf.l_margin + 2)
        pdf.set_font(_FONT, "B", 9.5)
        pdf.set_text_color(20, 20, 20)
        safe_link = entry.url if entry.url and entry.url.startswith(_LINK_PREFIXES) else ""
        label = printable(entry.technique)
        pdf.cell(pdf.get_string_width(label) + 2, 5, label, link=safe_link)
        details = f"{entry.title} - tactics: {', '.join(entry.tactics) or 'n/a'}" if entry.title else ""
        pdf.set_font(_FONT, size=9.5)
        pdf.multi_cell(0, 5, printable(details), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    heading("Evidence")
    bullets(list(content.evidence), empty="No evidence recorded.")

    heading("Remediation")
    actions = [f"{action}{' (requires human approval)' if gated else ''}" for action, gated in content.actions]
    bullets(actions, empty="No actions recommended." if not content.mitigations else "", marker="\N{BALLOT BOX}")
    if content.mitigations:
        pdf.ln(1)
        text("ATT&CK mitigations for the mapped techniques:", gray=True)
        bullets(list(content.mitigations), empty="")

    if content.notes:
        heading("Guardrail notes")
        bullets(list(content.notes), empty="")

    heading("Investigation trace")
    text(f"{len(content.steps)} steps, {content.input_tokens} input / {content.output_tokens} output tokens.")
    for number, step in enumerate(content.steps, start=1):
        line = "model produced a verdict" if step.tool is None else f"tool {step.tool} {step.arguments}"
        text(f"{number}. {line}", indent=2)
    return bytes(pdf.output())


def render_report_pdf(
    alert: dict[str, Any],
    investigation: Investigation,
    *,
    events: pd.DataFrame | None = None,
    retriever: HybridRetriever | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    """Build the PDF report with the same content as :func:`render_report`."""
    content = report_content(alert, investigation, events=events, retriever=retriever, generated_at=generated_at)
    return render_pdf(content)
