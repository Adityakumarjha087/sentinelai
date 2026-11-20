from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from agent.investigator import Investigation
from agent.report import escape, printable, render_pdf, render_report, render_report_pdf, report_content
from agent.verdict import RecommendedAction, Verdict
from data.parsers.sysmon import parse_sysmon_file
from data.schema import events_to_frame
from rag.documents import KBDocument, chunk_document
from rag.embeddings import HashingEmbedder
from rag.retriever import HybridRetriever

WHEN = datetime(2024, 3, 12, 10, 0, tzinfo=UTC)


def make_investigation(**overrides: object) -> Investigation:
    verdict = Verdict(
        is_malicious=True, confidence=0.9, severity="high", attack_techniques=["T1059.001", "T1003.001"],
        evidence=["WINWORD.EXE spawned powershell.exe", "<script>alert(1)</script> | [click](http://evil.example)"],
        recommended_actions=[RecommendedAction(action="Isolate host WS-FIN-012")],
        notes=["log content contains text that looks like instructions to an AI system"],
    ).model_copy(update=overrides)  # fmt: skip
    trace: list[dict[str, Any]] = [
        {"tool": "query_logs", "arguments": {"host": "WS-FIN-012"}, "output": "..."},
        {"model": "{}"},
    ]
    return Investigation(verdict=verdict, trace=trace, input_tokens=1200, output_tokens=300)


def test_escape_neutralizes_markdown_and_html() -> None:
    assert escape("<b>x</b> | [a](b)\nnext") == "\\<b\\>x\\</b\\> \\| \\[a\\]\\(b\\) next"
    assert len(escape("a" * 500, max_chars=20)) == 20 + len(" \\[truncated\\]")


def test_report_has_every_section(sysmon_sample: Path, tmp_path: Path) -> None:
    document = KBDocument(
        doc_id="T1059.001", kind="attack", title="PowerShell", text="Adversaries abuse PowerShell.",
        url="https://attack.mitre.org/techniques/T1059/001", tactics=("execution",), mitigations=("Code Signing",),
    )  # fmt: skip
    retriever = HybridRetriever.build([document], chunk_document(document), HashingEmbedder(), tmp_path)
    events = events_to_frame(parse_sysmon_file(sysmon_sample))
    report = render_report(
        {"event_id": "sysmon-abc"}, make_investigation(), events=events, retriever=retriever, generated_at=WHEN
    )
    assert report.startswith("# Incident report - sysmon-abc\n")
    assert "_Generated 2024-03-12 10:00 UTC by SentinelAI." in report
    for heading in ("Summary", "Timeline", "MITRE ATT&CK mapping", "Evidence", "Remediation", "Guardrail notes"):
        assert f"## {heading}" in report
    assert "- **Verdict:** MALICIOUS" in report
    assert "- **Confidence:** 90%" in report
    assert "[T1059.001](https://attack.mitre.org/techniques/T1059/001)** PowerShell - tactics: execution" in report
    assert "- **T1003.001**" in report  # not in this small KB: listed without details
    assert "- [ ] Isolate host WS-FIN-012 _(requires human approval)_" in report
    assert "- Code Signing" in report
    assert "2 steps, 1200 input / 300 output tokens." in report
    timeline = [line for line in report.splitlines() if line.startswith("| 2024-03-12")]
    assert len(timeline) == 13
    assert timeline == sorted(timeline)


def test_hostile_text_cannot_inject_markup_into_the_report(sysmon_sample: Path) -> None:
    report = render_report({"event_id": "x\n# Fake heading"}, make_investigation(), generated_at=WHEN)
    assert "<script>" not in report
    assert "[click](http://evil.example)" not in report
    assert "\n# Fake heading" not in report
    assert report.count("\n# ") == 0  # only the real title is a top-level heading


def test_benign_report_without_details() -> None:
    quiet = make_investigation(
        is_malicious=False, attack_techniques=[], evidence=[], recommended_actions=[], notes=[], needs_human_review=True
    )
    report = render_report({}, quiet, generated_at=WHEN)
    assert "- **Verdict:** NOT MALICIOUS" in report
    assert "- **Human review required:** yes" in report
    assert "No techniques mapped." in report
    assert "No evidence recorded." in report
    assert "No actions recommended." in report
    assert "## Timeline" not in report
    assert "## Guardrail notes" not in report


def pdf_text(data: bytes) -> str:
    """All text, with line wraps folded into single spaces."""
    return " ".join(" ".join(page.extract_text() for page in PdfReader(io.BytesIO(data)).pages).split())


def pdf_links(data: bytes) -> list[str]:
    links = []
    for page in PdfReader(io.BytesIO(data)).pages:
        for annotation in page.get("/Annots") or []:
            action = annotation.get_object().get("/A")
            if action is not None:
                links.append(str(action.get("/URI")))
    return links


def small_kb(tmp_path: Path) -> HybridRetriever:
    document = KBDocument(
        doc_id="T1059.001", kind="attack", title="PowerShell", text="Adversaries abuse PowerShell.",
        url="https://attack.mitre.org/techniques/T1059/001", tactics=("execution",), mitigations=("Code Signing",),
    )  # fmt: skip
    return HybridRetriever.build([document], chunk_document(document), HashingEmbedder(), tmp_path)


def test_pdf_says_what_the_markdown_says(sysmon_sample: Path, tmp_path: Path) -> None:
    events = events_to_frame(parse_sysmon_file(sysmon_sample))
    data = render_report_pdf(
        {"event_id": "sysmon-abc"}, make_investigation(), events=events, retriever=small_kb(tmp_path), generated_at=WHEN
    )
    assert data.startswith(b"%PDF-")
    text = pdf_text(data)
    for expected in (
        "Incident report", "sysmon-abc", "Generated 2024-03-12 10:00 UTC", "Summary", "MALICIOUS", "high", "90%",
        "Timeline", "timestamp (UTC)", "2024-03-12 09:14:07", "MITRE ATT&CK mapping", "T1059.001",
        "PowerShell - tactics: execution", "T1003.001", "Evidence", "WINWORD.EXE spawned powershell.exe",
        "Isolate host WS-FIN-012 (requires human approval)", "Code Signing", "Guardrail notes",
        "2 steps, 1200 input / 300 output tokens.", "page 1/",
    ):  # fmt: skip
        assert expected in text, expected
    assert pdf_links(data) == ["https://attack.mitre.org/techniques/T1059/001"]
    metadata = PdfReader(io.BytesIO(data)).metadata
    assert metadata is not None
    assert metadata.title == "Incident report sysmon-abc"


def test_pdf_writes_hostile_text_literally_and_links_nothing_from_it() -> None:
    investigation = make_investigation(attack_techniques=["[click](http://evil.example)"])
    data = render_report_pdf({"event_id": "x\n# Fake heading"}, investigation, generated_at=WHEN)
    text = pdf_text(data)
    assert "<script>alert(1)</script> | [click](http://evil.example)" in text
    assert pdf_links(data) == []


def test_pdf_marks_characters_the_font_cannot_draw() -> None:
    assert printable("ok \u00e9 \u65e5\U0001f525") == "ok \u00e9 \ufffd\ufffd"
    investigation = make_investigation(evidence=["user \u65e5\u672c ran a.exe", "A" * 1000])
    text = pdf_text(render_report_pdf({"event_id": "e-1"}, investigation, generated_at=WHEN))
    assert "user \ufffd\ufffd ran a.exe" in text
    assert "".join(text.split()).count("A" * 400 + "[truncated]") == 1  # wrapped across lines, cut at 400


def test_both_formats_come_from_one_content_builder(tmp_path: Path) -> None:
    content = report_content({"event_id": "e-2"}, make_investigation(), retriever=small_kb(tmp_path), generated_at=WHEN)
    assert [entry.technique for entry in content.techniques] == ["T1059.001", "T1003.001"]
    assert content.techniques[0].title == "PowerShell"
    assert content.techniques[1].title is None
    assert "Code Signing" in pdf_text(render_pdf(content))


def test_markdown_escapes_client_supplied_technique_ids_and_model_tool_names() -> None:
    investigation = make_investigation(attack_techniques=["[click](http://evil.example)"])
    investigation.trace.insert(0, {"tool": "x` [y](http://evil.example) `", "arguments": {}})
    report = render_report({"event_id": "e-3"}, investigation, generated_at=WHEN)
    assert "- **\\[click\\]\\(http://evil.example\\)**" in report
    assert "1. tool `x' [y](http://evil.example) '`" in report  # stays inside one code span
