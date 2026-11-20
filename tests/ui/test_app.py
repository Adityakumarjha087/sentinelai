from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "ui" / "app.py")


def test_dashboard_renders_feed_and_investigates_an_alert() -> None:
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value.startswith("SentinelAI")
    assert len(app.dataframe[0].value) == 13
    assert app.info[0].value == "Not investigated yet."

    app.button[1].click().run()  # "Investigate" for the selected alert
    assert not app.exception
    assert [metric.label for metric in app.metric] == ["Verdict", "Severity", "Confidence"]
    assert not app.info


def test_investigate_all_flags_the_injected_alert_for_review() -> None:
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.button[0].click().run()  # "Investigate all"
    assert not app.exception
    results = app.session_state["results"]
    assert len(results) == 13
    flagged = [r for r in results.values() if any("instructions to an AI" in n for n in r["verdict"]["notes"])]
    assert len(flagged) == 1
