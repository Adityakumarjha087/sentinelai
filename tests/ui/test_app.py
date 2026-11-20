from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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


def test_dashboard_renders_feed_and_investigates_an_alert() -> None:
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value.startswith("SentinelAI")
    assert len(app.dataframe[0].value) == 13
    assert "Not investigated yet." in infos(app)
    assert any(text.startswith("Flow detector not loaded: not built") for text in infos(app))
    assert "Knowledge base: not built" in app.caption[0].value

    app.button(key="host-one").click().run()
    assert not app.exception
    assert [metric.label for metric in app.metric] == ["Verdict", "Severity", "Confidence"]
    assert "Not investigated yet." not in infos(app)
    (result,) = app.session_state["results"].values()
    assert result["report_pdf"].startswith(b"%PDF-")  # rendered for the download button


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
    assert "Flow detector: isolation_forest" in app.caption[0].value
    detections = app.dataframe[1].value
    assert len(detections) == 30
    assert "ground_truth" in detections.columns
    assert "never to the analyst" in app.caption[1].value

    app.button(key="flow-all").click().run()
    assert not app.exception
    results = app.session_state["results"]
    assert len(results) == 30  # only the flow alerts were investigated
    scans = [r for r in results.values() if r["verdict"]["attack_techniques"] == ["T1046"]]
    assert scans
    assert all("PortScan" not in r["report_markdown"] for r in results.values())
