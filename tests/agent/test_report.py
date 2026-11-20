from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.investigator import Investigation
from agent.report import escape, render_report
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
