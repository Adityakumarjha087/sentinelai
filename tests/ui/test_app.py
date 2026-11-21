from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from common.settings import get_settings
from tests.conftest import FlowArtifacts

APP = str(Path(__file__).resolve().parents[2] / "ui" / "app.py")


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    """The dashboard caches its resources per process; each test sets up its own."""
    st.cache_resource.clear()
    get_settings.cache_clear()
    yield
    st.cache_resource.clear()
    get_settings.cache_clear()


def infos(app: AppTest) -> list[str]:
    return [str(info.value) for info in app.info]


def captions(app: AppTest) -> list[str]:
    return [str(caption.value) for caption in app.caption]


def sidebar_status(app: AppTest) -> dict[str, str]:
    """Component name -> detail, as the sidebar lists them."""
    names = [str(md.value).split("**")[1] for md in app.sidebar.markdown]
    details = [str(caption.value) for caption in app.sidebar.caption]
    return dict(zip(names, details, strict=False))


def metrics(app: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in app.metric}


def test_dashboard_renders_feed_and_investigates_an_alert() -> None:
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value.startswith("SentinelAI")
    assert len(app.dataframe[0].value) == 13
    assert "Not investigated yet." in infos(app)
    assert any(text.startswith("Flow detector not loaded: not built") for text in infos(app))
    status = sidebar_status(app)
    assert status["Knowledge base"].startswith("not built")
    assert status["Injection detection"] == "Patterns only"
    assert "No LLM is configured" in status["Analyst"]
    assert metrics(app) == {
        "Host events": "13",
        "Network detections": "-",
        "Investigated": "0",
        "Malicious": "0",
        "Needs review": "0",
    }

    app.button(key="host-one").click().run()
    assert not app.exception
    assert "Not investigated yet." not in infos(app)
    (result,) = app.session_state["results"].values()
    assert result["report_pdf"].startswith(b"%PDF-")  # rendered for the download button
    assert metrics(app)["Investigated"] == "1"  # counted in the same rerun as the click
    badges = [str(md.value) for md in app.markdown if "-badge[" in str(md.value)]
    assert any("alicious]" in badge for badge in badges)
    assert any(f"{result['verdict']['severity']} severity]" in badge for badge in badges)
    assert app.dataframe[0].value["verdict"].iloc[0] != ""  # the feed shows the new verdict


def test_verdict_card_lists_evidence_and_actions_needing_approval() -> None:
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key="host-all").click().run()
    lsass = next(i for i, option in enumerate(app.selectbox(key="host-pick").options) if "process_access" in option)
    app.selectbox(key="host-pick").set_value(lsass).run()
    assert not app.exception
    text = "\n".join(str(md.value) for md in app.markdown)
    assert "**Evidence**" in text
    assert "lsass" in text
    assert "**Recommended actions**" in text
    assert "needs approval]" in text


def test_investigate_all_flags_the_injected_alert_for_review() -> None:
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button(key="host-all").click().run()
    assert not app.exception
    results = app.session_state["results"]
    assert len(results) == 13
    flagged = [r for r in results.values() if any("instructions to an AI" in n for n in r["verdict"]["notes"])]
    assert len(flagged) == 1


def test_network_tab_investigates_detections_from_a_built_detector(
    flow_artifacts: FlowArtifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SENTINEL_DETECTOR_DIR", str(flow_artifacts.model_dir))
    monkeypatch.setenv("SENTINEL_FEATURIZER_PATH", str(flow_artifacts.featurizer_path))
    monkeypatch.setenv("SENTINEL_DEMO_FLOWS_PATH", str(flow_artifacts.flows_path))
    monkeypatch.setenv("SENTINEL_MAX_DETECTIONS", "30")
    get_settings.cache_clear()

    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert sidebar_status(app)["Flow detector"] == "isolation_forest"
    assert metrics(app)["Network detections"] == "30"
    detections = app.dataframe[1].value
    assert len(detections) == 30
    assert "ground_truth" in detections.columns
    assert any("never to the analyst" in caption for caption in captions(app))

    app.button(key="flow-all").click().run()
    assert not app.exception
    results = app.session_state["results"]
    assert len(results) == 30  # only the flow alerts were investigated
    scans = [r for r in results.values() if r["verdict"]["attack_techniques"] == ["T1046"]]
    assert scans
    assert metrics(app)["Investigated"] == "30"
    assert all("PortScan" not in r["report_markdown"] for r in results.values())


def test_technique_links_only_well_formed_ids() -> None:
    from ui.app import technique_markdown

    assert technique_markdown("T1046") == "[T1046](https://attack.mitre.org/techniques/T1046/)"
    assert technique_markdown("T1059.001") == "[T1059.001](https://attack.mitre.org/techniques/T1059/001/)"
    assert technique_markdown("[x](https://evil.example)") == "`[x](https://evil.example)`"
    assert technique_markdown("T1`046") == "`T1046`"


def test_inert_escapes_markdown_links_and_directives() -> None:
    from ui.app import inert

    rendered = inert("see [here](https://evil.example) :red[alarm] ![x](y.png)\n# heading")
    assert "[here](" not in rendered
    assert ":red[" not in rendered
    assert "![x]" not in rendered
    assert "\n" not in rendered  # no new block (heading, list) can start
    assert rendered.replace("\\", "") == "see [here](https://evil.example) :red[alarm] ![x](y.png) # heading"


def _render_trace() -> None:
    from ui.app import show_trace

    show_trace(
        [
            {"tool": "search_logs", "arguments": {"host": "WS-1"}, "output": '{"events": [1, 2]}'},
            {"tool": "lookup`x", "arguments": {}, "output": "plain text, not json"},
            {"model": '{"is_malicious": true}'},
        ]
    )


def test_trace_renders_numbered_tool_calls_and_replies() -> None:
    app = AppTest.from_function(_render_trace).run()
    assert not app.exception
    headings = [str(md.value) for md in app.markdown]
    assert headings == ["**1. Tool call** · `search_logs`", "**2. Tool call** · `lookupx`", "**3. Analyst reply**"]
    blocks = [(block.value, block.language) for block in app.code]
    assert ('{\n  "events": [\n    1,\n    2\n  ]\n}', "json") in blocks  # tool output pretty-printed
    assert any(value == "plain text, not json" and language != "json" for value, language in blocks)
    assert any(value == '{\n  "is_malicious": true\n}' for value, _ in blocks)


def test_severity_chart_counts_every_level_in_order() -> None:
    from ui.app import SEVERITIES, severity_chart

    results = [{"verdict": {"severity": level}} for level in ["high", "low", "high", "critical"]]
    chart = severity_chart(results)
    frame = chart.data
    assert list(frame["severity"]) == SEVERITIES
    assert list(frame["alerts"]) == [1, 0, 2, 1]
    assert chart.to_dict()["encoding"]["color"]["scale"]["domain"] == SEVERITIES


def test_anomaly_scores_are_bars_scaled_to_the_highest_score() -> None:
    from ui.app import feed_column_config

    config = feed_column_config(pd.DataFrame({"anomaly_score": [0.1, 0.4, 0.25]}))
    bar = config["anomaly_score"]
    assert bar["type_config"]["type"] == "progress"
    assert bar["type_config"]["max_value"] == 0.4
    assert "anomaly_score" not in feed_column_config(pd.DataFrame({"host": ["a"]}))
