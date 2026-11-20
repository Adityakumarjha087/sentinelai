"""SentinelAI dashboard: host and network alert feeds, investigation trace, severity chart, report download.

Run with ``make ui`` (``streamlit run ui/app.py``). The network tab and the
knowledge base light up when they are built (``make data train`` and ``make kb``).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import alert_from_row, run_investigation
from api.resources import Resources, load_resources
from detection.service import GROUND_TRUTH, flow_alert
from guardrails.injection import detect_injection

HOST_COLUMNS = ["timestamp", "event_type", "host", "process", "dst_ip", "injection_suspected"]
FLOW_COLUMNS = ["timestamp", "src_ip", "dst_ip", "dst_port", "anomaly_score", GROUND_TRUTH]
SEVERITIES = ["low", "medium", "high", "critical"]


@st.cache_resource
def resources() -> Resources:
    """Sample events plus whichever optional artifacts are built, loaded once per process."""
    return load_resources()


def host_events(events: pd.DataFrame) -> pd.DataFrame:
    """Host events, newest first, each flagged if it looks like a prompt injection."""
    frame = events.copy()
    text = frame.astype(object).fillna("").astype(str).agg(" ".join, axis=1)
    frame["injection_suspected"] = [detect_injection(row).is_suspicious for row in text]
    return frame.sort_values("timestamp", ascending=False).reset_index(drop=True)


def status_line(loaded: Resources) -> str:
    """What this dashboard is running with, stated plainly."""
    kb = "loaded" if loaded.retriever else loaded.unavailable.get("knowledge_base", "not loaded")
    detector = loaded.feed.summary["detector"] if loaded.feed else loaded.unavailable.get("detector", "not loaded")
    return (
        "Host events: a synthetic Sysmon sample. No LLM is configured, so investigations use the offline "
        f"rule-based analyst - a baseline, not a language model. Knowledge base: {kb}. Flow detector: {detector}."
    )


def show_result(result: dict[str, Any] | None, alert: dict[str, Any]) -> None:
    """Verdict, guardrail notes, trace and report for one investigated alert."""
    with st.expander("What the analyst sees"):
        st.json(alert)
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
        "Download report (.md)",
        result["report_markdown"],
        file_name=f"incident-{alert['event_id']}.md",
        key=f"download-{alert['event_id']}",
    )


def alert_panel(
    key: str,
    table: pd.DataFrame,
    columns: list[str],
    to_alert: Callable[[pd.Series], dict[str, Any]],
    describe: Callable[[pd.Series], str],
) -> None:
    """Feed, bulk and single investigation, and severity chart for one alert source."""
    if table.empty:
        st.info("No alerts.")
        return
    loaded = resources()
    events = loaded.investigation_events()
    results: dict[str, dict[str, Any]] = st.session_state.setdefault("results", {})

    feed, detail = st.columns([3, 2])
    with feed:
        st.dataframe(table[[column for column in columns if column in table]], hide_index=True, width="stretch")
        if st.button("Investigate all", key=f"{key}-all"):
            for _, row in table.iterrows():
                alert = to_alert(row)
                results[alert["event_id"]] = run_investigation(alert, events, loaded.retriever, None)
        done = [results[event_id] for event_id in table["event_id"] if event_id in results]
        if done:
            st.subheader("Severity of investigated alerts")
            counts = pd.Series([result["verdict"]["severity"] for result in done]).value_counts()
            st.bar_chart(counts.reindex(SEVERITIES, fill_value=0), x_label="severity", y_label="alerts")

    with detail:
        st.subheader("Investigation")
        rows = [row for _, row in table.iterrows()]
        choice = st.selectbox(
            "Alert", range(len(rows)), format_func=lambda index: describe(rows[index]), key=f"{key}-pick"
        )
        alert = to_alert(rows[int(choice or 0)])
        if st.button("Investigate", key=f"{key}-one"):
            results[alert["event_id"]] = run_investigation(alert, events, loaded.retriever, None)
        show_result(results.get(alert["event_id"]), alert)


def network_caption(summary: dict[str, Any], shown: int) -> str:
    share = summary.get("flagged_attack_share")
    truth = f" {share:.1%} of flagged flows are attacks by the dataset's labels." if share is not None else ""
    return (
        f"The {summary['detector']} detector scored {summary['flows_scored']:,} flows from {summary['source']} and "
        f"flagged {summary['flows_flagged']:,} (threshold {summary['threshold']:.4f}).{truth} Showing a random "
        f"sample of {shown}, oldest first. `{GROUND_TRUTH}` is that label: shown here, never to the analyst."
    )


def main() -> None:
    st.set_page_config(page_title="SentinelAI", layout="wide")
    st.title("SentinelAI - AI SOC analyst")
    loaded = resources()
    st.caption(status_line(loaded))

    host_tab, network_tab = st.tabs(["Host events", "Network detections"])
    with host_tab:
        alert_panel(
            "host",
            host_events(loaded.events),
            HOST_COLUMNS,
            alert_from_row,
            lambda row: f"{row['timestamp']:%H:%M:%S}  {row['event_type']}  {row['event_id']}",
        )
    with network_tab:
        if loaded.feed is None:
            st.info(f"Flow detector not loaded: {loaded.unavailable.get('detector', 'unknown reason')}.")
        else:
            detections = loaded.feed.detections
            st.caption(network_caption(loaded.feed.summary, len(detections)))
            alert_panel(
                "flow",
                detections,
                FLOW_COLUMNS,
                flow_alert,
                lambda row: (
                    f"{row['timestamp']:%H:%M:%S}  {row['src_ip']} -> {row['dst_ip']}:{row['dst_port']}  "
                    f"score {row['anomaly_score']:.3f}"
                ),
            )


main()
