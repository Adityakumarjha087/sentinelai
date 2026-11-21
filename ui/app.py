"""SentinelAI dashboard: host and network alert feeds, investigation trace, severity chart, report download.

Run with ``make ui`` (``streamlit run ui/app.py``). The network tab and the
knowledge base light up when they are built (``make data train`` and ``make kb``).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.investigator import Investigation
from agent.report import render_report_pdf
from agent.verdict import Verdict
from api.main import alert_from_row, report_filename, run_investigation
from api.resources import Resources, load_resources
from detection.service import GROUND_TRUTH, flow_alert
from guardrails.injection import detect_injection
from rag.retriever import HybridRetriever

HOST_COLUMNS = ["verdict", "timestamp", "event_type", "host", "process", "dst_ip", "injection_suspected"]
FLOW_COLUMNS = ["verdict", "timestamp", "src_ip", "dst_ip", "dst_port", "anomaly_score", GROUND_TRUTH]
SEVERITIES = ["low", "medium", "high", "critical"]
FEED_FILTERS = ["All", "Not investigated", "Malicious", "Needs review"]
SEVERITY_COLORS: dict[str, Literal["blue", "yellow", "orange", "red"]] = {
    "low": "blue",
    "medium": "yellow",
    "high": "orange",
    "critical": "red",
}
# Bar colours for the severity chart, in step with the badge colours above.
SEVERITY_HEX = {"low": "#3b82f6", "medium": "#eab308", "high": "#f97316", "critical": "#ef4444"}
# ATT&CK ids come from the analyst's output; only well-formed ones become links.
TECHNIQUE_RE = re.compile(r"^T(\d{4})(?:\.(\d{3}))?$")
# Markdown (and Streamlit directive) punctuation, escaped in text the analyst wrote.
MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+\-.!|~$:])")
FEED_COLUMN_CONFIG = {
    "verdict": st.column_config.TextColumn(
        "verdict", width="medium", pinned=True, help="Outcome once investigated (severity in brackets)"
    ),
    "timestamp": st.column_config.DatetimeColumn("timestamp", format="YYYY-MM-DD HH:mm:ss"),
    "injection_suspected": st.column_config.CheckboxColumn(
        "injection?", help="Text in the event looks like instructions to an AI"
    ),
    GROUND_TRUTH: st.column_config.TextColumn(GROUND_TRUTH, help="Dataset label: shown here, never to the analyst"),
}


@dataclass(frozen=True)
class Component:
    """One part of the deployment and whether it is running. ``ready=None`` is informational."""

    name: str
    detail: str
    ready: bool | None


@st.cache_resource
def resources() -> Resources:
    """Sample events plus whichever optional artifacts are built, loaded once per process."""
    return load_resources()


def feed_column_config(table: pd.DataFrame) -> dict[str, Any]:
    """Column display settings; anomaly scores become bars scaled to the highest score in view."""
    config: dict[str, Any] = dict(FEED_COLUMN_CONFIG)
    if "anomaly_score" in table and not table.empty:
        config["anomaly_score"] = st.column_config.ProgressColumn(
            "anomaly score",
            format="%.3f",
            min_value=0.0,
            max_value=max(float(table["anomaly_score"].max()), 1e-9),
            help="Detector score; higher is more unusual. Bars are relative to the highest score shown.",
        )
    return config


def results_store() -> dict[str, dict[str, Any]]:
    """Investigation results for this browser session, keyed by event id."""
    results: dict[str, dict[str, Any]] = st.session_state.setdefault("results", {})
    return results


def investigate(alerts: list[dict[str, Any]]) -> None:
    """Button callback: runs before the rerun, so every panel draws the new results."""
    loaded = resources()
    events = loaded.investigation_events()
    results = results_store()
    for alert in alerts:
        results[alert["event_id"]] = run_investigation(alert, events, loaded.retriever, None)


def host_events(events: pd.DataFrame) -> pd.DataFrame:
    """Host events, newest first, each flagged if it looks like a prompt injection."""
    frame = events.copy()
    text = frame.astype(object).fillna("").astype(str).agg(" ".join, axis=1)
    frame["injection_suspected"] = [detect_injection(row).is_suspicious for row in text]
    return frame.sort_values("timestamp", ascending=False).reset_index(drop=True)


def system_status(loaded: Resources) -> list[Component]:
    """What this dashboard is running with, one line per component."""
    detector = loaded.feed.summary["detector"] if loaded.feed else None
    return [
        Component("Host events", "Synthetic Sysmon sample", True),
        Component(
            "Analyst",
            "Offline rule-based baseline. No LLM is configured, so this is not a language model.",
            None,
        ),
        Component(
            "Knowledge base",
            "ATT&CK + CVE, loaded" if loaded.retriever else loaded.unavailable.get("knowledge_base", "not loaded"),
            loaded.retriever is not None,
        ),
        Component(
            "Flow detector",
            detector or loaded.unavailable.get("detector", "not loaded"),
            detector is not None,
        ),
        Component(
            "Injection detection",
            "Patterns + fine-tuned DistilBERT" if loaded.injection_classifier else "Patterns only",
            loaded.injection_classifier is not None,
        ),
    ]


def sidebar(loaded: Resources) -> None:
    with st.sidebar:
        st.subheader("System status", anchor=False)
        for component in system_status(loaded):
            icon = {
                True: ":green[:material/check_circle:]",
                False: ":orange[:material/error:]",
                None: ":blue[:material/info:]",
            }[component.ready]
            st.markdown(f"{icon} **{component.name}**")
            st.caption(component.detail)
        st.divider()
        st.caption(
            "Investigations run in this process with the offline analyst. Anything that would change a "
            "system is left as a recommendation for a person to approve."
        )


def overview(loaded: Resources, results: dict[str, dict[str, Any]]) -> None:
    """Headline counts across both feeds."""
    verdicts = [result["verdict"] for result in results.values()]
    columns = st.columns(5)
    columns[0].metric("Host events", len(loaded.events), border=True)
    columns[1].metric("Network detections", len(loaded.feed.detections) if loaded.feed else "-", border=True)
    columns[2].metric("Investigated", len(verdicts), border=True)
    columns[3].metric("Malicious", sum(v["is_malicious"] for v in verdicts), border=True)
    columns[4].metric("Needs review", sum(v["needs_human_review"] for v in verdicts), border=True)


def inert(text: str) -> str:
    """Analyst-written text for st.markdown, shown as typed: no links, images, colours or formatting."""
    return MARKDOWN_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))


def technique_markdown(technique: str) -> str:
    """A link to the technique's ATT&CK page, or the id as inert code if it is not well formed."""
    match = TECHNIQUE_RE.match(technique)
    if not match:
        return f"`{technique.replace('`', '')}`"
    path = f"T{match[1]}/{match[2]}" if match[2] else f"T{match[1]}"
    return f"[{technique}](https://attack.mitre.org/techniques/{path}/)"


def filter_feed(table: pd.DataFrame, results: dict[str, dict[str, Any]], choice: str | None) -> pd.DataFrame:
    """The rows of ``table`` that match a feed filter; ``None`` (nothing picked) means all."""
    if choice in (None, "All"):
        return table

    def keep(event_id: str) -> bool:
        result = results.get(event_id)
        if choice == "Not investigated":
            return result is None
        if result is None:
            return False
        return bool(result["verdict"]["is_malicious" if choice == "Malicious" else "needs_human_review"])

    return table[[keep(event_id) for event_id in table["event_id"]]]


def picked_index(selected_rows: list[int], shown_ids: list[str], all_ids: list[str]) -> int | None:
    """Position in the full feed of the row selected in the (possibly filtered) table."""
    if not selected_rows or selected_rows[0] >= len(shown_ids):
        return None
    return all_ids.index(shown_ids[selected_rows[0]])


def select_row(key: str, table_key: str, shown_ids: list[str], all_ids: list[str]) -> None:
    """Table callback: open the clicked alert in the investigation panel."""
    index = picked_index(list(st.session_state[table_key].selection.rows), shown_ids, all_ids)
    if index is not None:
        st.session_state[f"{key}-pick"] = index


def verdict_label(result: dict[str, Any] | None) -> str:
    if result is None:
        return ""
    verdict = result["verdict"]
    outcome = "malicious" if verdict["is_malicious"] else "not malicious"
    return f"{outcome} ({verdict['severity']})"


def severity_chart(results: list[dict[str, Any]]) -> alt.Chart:
    """Investigated alerts per severity, every level shown, bars coloured like the severity badges."""
    counts = pd.Series([result["verdict"]["severity"] for result in results]).value_counts()
    frame = pd.DataFrame({"severity": SEVERITIES, "alerts": counts.reindex(SEVERITIES, fill_value=0).to_numpy()})
    chart: alt.Chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("severity:N", sort=SEVERITIES, title="severity", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("alerts:Q", title="alerts", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(domain=SEVERITIES, range=[SEVERITY_HEX[level] for level in SEVERITIES]),
                legend=None,
            ),
            tooltip=["severity:N", "alerts:Q"],
        )
        .properties(height=260)
    )
    return chart


def report_pdf(result: dict[str, Any], alert: dict[str, Any], retriever: HybridRetriever | None) -> bytes:
    """The PDF report for one result, rendered on first use and kept with the result."""
    if "report_pdf" not in result:
        investigation = Investigation(
            verdict=Verdict.model_validate(result["verdict"]), trace=result["trace"], **result["usage"]
        )
        result["report_pdf"] = render_report_pdf(alert, investigation, retriever=retriever)
    pdf: bytes = result["report_pdf"]
    return pdf


def show_verdict(verdict: dict[str, Any]) -> None:
    """Outcome, severity and review flag as badges, then confidence, ATT&CK, evidence and actions."""
    with st.container(border=True):
        with st.container(horizontal=True, gap="small"):
            if verdict["is_malicious"]:
                st.badge("Malicious", icon=":material/gpp_bad:", color="red")
            else:
                st.badge("Not malicious", icon=":material/verified_user:", color="green")
            st.badge(f"{verdict['severity']} severity", color=SEVERITY_COLORS[verdict["severity"]])
            if verdict["needs_human_review"]:
                st.badge("Needs human review", icon=":material/flag:", color="orange")
        st.progress(verdict["confidence"], text=f"Confidence {verdict['confidence']:.0%}")
        techniques = " · ".join(technique_markdown(t) for t in verdict["attack_techniques"])
        st.markdown(f"**ATT&CK** {techniques or 'none mapped'}")
        if verdict["evidence"]:
            st.markdown("**Evidence**\n" + "\n".join(f"- {inert(item)}" for item in verdict["evidence"]))
        if verdict["recommended_actions"]:
            st.markdown(
                "**Recommended actions**\n"
                + "\n".join(
                    f"- {inert(action['action'])}"
                    + (" :orange-badge[:material/lock: needs approval]" if action["requires_human_approval"] else "")
                    for action in verdict["recommended_actions"]
                )
            )


def pretty(text: str) -> tuple[str, str | None]:
    """Pretty-printed JSON and its language if ``text`` parses as JSON, else the text unchanged."""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False), "json"
    except (TypeError, ValueError):
        return text, None


def show_trace(trace: list[dict[str, Any]]) -> None:
    """Each step in order: tool calls with arguments and output, and the analyst's replies."""
    for number, step in enumerate(trace, start=1):
        if "tool" in step:
            st.markdown(f"**{number}. Tool call** · `{str(step['tool']).replace('`', '')}`")
            st.code(json.dumps(step.get("arguments", {}), indent=2, default=str), language="json")
            body, language = pretty(str(step.get("output", "")))
            st.caption("Output")
            st.code(body, language=language, wrap_lines=True, height=320 if body.count("\n") > 14 else "content")
        else:
            st.markdown(f"**{number}. Analyst reply**")
            body, language = pretty(str(step.get("model", "")))
            st.code(body, language=language, wrap_lines=True)


def show_result(result: dict[str, Any] | None, alert: dict[str, Any], retriever: HybridRetriever | None) -> None:
    """Verdict, guardrail notes, trace and report for one investigated alert."""
    if result is None:
        st.info("Not investigated yet.", icon=":material/hourglass_empty:")
    else:
        show_verdict(result["verdict"])
        for note in result["verdict"]["notes"]:
            st.error(f"Guardrail: {inert(note)}", icon=":material/shield:")
    with st.expander("What the analyst sees"):
        st.json(alert)
    if result is None:
        return
    with st.expander(f"Reasoning trace ({len(result['trace'])} steps, analyst: {result['analyst']})"):
        steps_tab, json_tab = st.tabs(["Steps", "JSON"])
        with steps_tab:
            show_trace(result["trace"])
        with json_tab:
            st.json(result["trace"])
    with st.expander("Incident report"):
        st.markdown(result["report_markdown"])
    with st.container(horizontal=True, gap="small"):
        st.download_button(
            "Report (.md)",
            result["report_markdown"],
            file_name=report_filename(alert, "md"),
            mime="text/markdown",
            icon=":material/download:",
            key=f"download-md-{alert['event_id']}",
        )
        st.download_button(
            "Report (.pdf)",
            report_pdf(result, alert, retriever),
            file_name=report_filename(alert, "pdf"),
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
            key=f"download-pdf-{alert['event_id']}",
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
    results = results_store()
    rows = [row for _, row in table.iterrows()]

    feed, detail = st.columns([3, 2], gap="large")
    with feed:
        done = [results[event_id] for event_id in table["event_id"] if event_id in results]
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{len(rows)} alerts** · {len(done)} investigated")
            st.button(
                "Investigate all",
                key=f"{key}-all",
                icon=":material/manage_search:",
                on_click=lambda: investigate([to_alert(row) for row in rows]),
            )
        feed_filter = st.segmented_control(
            "Show", FEED_FILTERS, default="All", key=f"{key}-filter", label_visibility="collapsed"
        )
        shown = filter_feed(table, results, feed_filter)
        shown = shown.assign(verdict=[verdict_label(results.get(event_id)) for event_id in shown["event_id"]])
        # One table key per filter: a selection made in one view never ticks a different row in another.
        table_key = f"{key}-table-{feed_filter or 'All'}"
        if shown.empty:
            st.caption("No alerts match this filter.")
        else:
            st.dataframe(
                shown[[column for column in columns if column in shown]],
                hide_index=True,
                width="stretch",
                column_config=feed_column_config(table),
                key=table_key,
                on_select=partial(select_row, key, table_key, list(shown["event_id"]), list(table["event_id"])),
                selection_mode="single-row",
            )
            st.caption("Select a row to open it under Investigation.")
        if done:
            st.subheader("Severity of investigated alerts", anchor=False)
            st.altair_chart(severity_chart(done), width="stretch")

    with detail:
        st.subheader("Investigation", anchor=False)
        choice = st.selectbox(
            "Alert", range(len(rows)), format_func=lambda index: describe(rows[index]), key=f"{key}-pick"
        )
        alert = to_alert(rows[int(choice or 0)])
        st.button(
            "Investigate",
            key=f"{key}-one",
            type="primary",
            icon=":material/search:",
            on_click=investigate,
            args=([alert],),
        )
        show_result(results.get(alert["event_id"]), alert, loaded.retriever)


def network_caption(summary: dict[str, Any], shown: int) -> str:
    share = summary.get("flagged_attack_share")
    truth = f" {share:.1%} of flagged flows are attacks by the dataset's labels." if share is not None else ""
    return (
        f"The {summary['detector']} detector scored {summary['flows_scored']:,} flows from {summary['source']} and "
        f"flagged {summary['flows_flagged']:,} (threshold {summary['threshold']:.4f}).{truth} Showing a random "
        f"sample of {shown}, oldest first. `{GROUND_TRUTH}` is that label: shown here, never to the analyst."
    )


def main() -> None:
    st.set_page_config(page_title="SentinelAI", page_icon=":material/shield:", layout="wide")
    loaded = resources()
    st.title("SentinelAI", anchor=False)
    st.caption("AI SOC analyst: triage host and network alerts, map them to MITRE ATT&CK, write the incident report.")
    sidebar(loaded)
    overview(loaded, results_store())

    host_tab, network_tab = st.tabs([":material/computer: Host events", ":material/lan: Network detections"])
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


if __name__ == "__main__":  # streamlit runs the file as __main__; tests import helpers
    main()
