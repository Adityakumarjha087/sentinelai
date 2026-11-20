"""SentinelAI dashboard: alert feed, investigation trace, severity chart, one-click report.

Run with ``make ui`` (``streamlit run ui/app.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import run_investigation
from common.settings import get_settings
from data.parsers.sysmon import parse_sysmon_file
from data.schema import events_to_frame
from guardrails.injection import detect_injection
from guardrails.sanitize import sanitize_text

FEED_COLUMNS = ["timestamp", "event_type", "host", "process", "dst_ip", "injection_suspected"]


@st.cache_data
def load_events() -> pd.DataFrame:
    """Events from the bundled synthetic Sysmon sample."""
    samples = sorted(get_settings().samples_dir.glob("*.xml"))
    frame = events_to_frame([event for path in samples for event in parse_sysmon_file(path)])
    text = frame.astype(object).fillna("").astype(str).agg(" ".join, axis=1)
    frame["injection_suspected"] = [detect_injection(row).is_suspicious for row in text]
    return frame.sort_values("timestamp", ascending=False).reset_index(drop=True)


def alert_from_row(row: pd.Series) -> dict[str, str]:
    """The fields of one event that are worth showing to the analyst."""
    return {str(key): sanitize_text(value, 600) for key, value in row.items() if key != "raw" and not pd.isna(value)}


def main() -> None:
    st.set_page_config(page_title="SentinelAI", layout="wide")
    st.title("SentinelAI - AI SOC analyst")
    st.caption(
        "Demo data: a synthetic Sysmon sample. No LLM is configured, so investigations use the "
        "offline rule-based analyst - a baseline, not a language model."
    )
    events = load_events()
    results: dict[str, dict[str, Any]] = st.session_state.setdefault("results", {})

    feed, detail = st.columns([3, 2])
    with feed:
        st.subheader("Alert feed")
        st.dataframe(events[FEED_COLUMNS], hide_index=True, use_container_width=True)
        if st.button("Investigate all"):
            for _, row in events.iterrows():
                alert = alert_from_row(row)
                results[alert["event_id"]] = run_investigation(alert, events, None, None)
        if results:
            st.subheader("Severity of investigated alerts")
            order = ["low", "medium", "high", "critical"]
            counts = pd.Series([r["verdict"]["severity"] for r in results.values()]).value_counts()
            st.bar_chart(counts.reindex(order, fill_value=0), x_label="severity", y_label="alerts")

    with detail:
        st.subheader("Investigation")
        labels = [f"{row.timestamp:%H:%M:%S}  {row.event_type}  {row.event_id}" for row in events.itertuples()]
        choice = st.selectbox("Alert", range(len(events)), format_func=lambda index: labels[index])
        alert = alert_from_row(events.iloc[choice])
        if st.button("Investigate"):
            results[alert["event_id"]] = run_investigation(alert, events, None, None)
        result = results.get(alert["event_id"])
        if result is None:
            st.info("Not investigated yet.")
            return
        verdict = result["verdict"]
        left, middle, right = st.columns(3)
        left.metric("Verdict", "Malicious" if verdict["is_malicious"] else "Not malicious")
        middle.metric("Severity", verdict["severity"])
        right.metric("Confidence", f"{verdict['confidence']:.0%}")
        if verdict["needs_human_review"]:
            st.warning("Human review required.")
        for note in verdict["notes"]:
            st.error(f"Guardrail: {note}")
        st.write("**ATT&CK:**", ", ".join(verdict["attack_techniques"]) or "none mapped")
        with st.expander(f"Reasoning trace ({len(result['trace'])} steps, analyst: {result['analyst']})"):
            st.json(result["trace"])
        with st.expander("Incident report"):
            st.markdown(result["report_markdown"])
        st.download_button(
            "Download report (.md)", result["report_markdown"], file_name=f"incident-{alert['event_id']}.md"
        )


main()
