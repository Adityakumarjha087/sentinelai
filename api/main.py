"""Read-only API over the event store, flow detections, knowledge base and reports.

``/investigate`` uses whichever analyst the app was created with. Without a
configured LLM that is the offline rule-based baseline, and responses say so.
The knowledge base and flow detector are optional; ``/health`` says which are
loaded and why any are not.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict

from agent.investigator import Investigation, investigate
from agent.llm import LLMClient
from agent.report import render_report, render_report_pdf
from agent.tools import ToolBox
from agent.verdict import Verdict
from api.resources import Resources, load_resources
from detection.service import PEOPLE_ONLY, DetectionFeed, flow_alert
from eval.analysts import RuleBasedAnalyst
from guardrails.injection import detect_injection
from guardrails.sanitize import sanitize_text
from rag.retriever import HybridRetriever

_ALERT_FIELDS = ("event_id", "timestamp", "event_type", "host", "process", "user", "src_ip", "dst_ip", "dst_port")


class ReportRequest(BaseModel):
    """An alert and the verdict reached for it."""

    model_config = ConfigDict(extra="forbid")

    alert: dict[str, Any]
    verdict: Verdict


class InvestigateRequest(BaseModel):
    """An alert to investigate."""

    model_config = ConfigDict(extra="forbid")

    alert: dict[str, Any]


def alert_from_row(row: pd.Series) -> dict[str, str]:
    """Every populated field of one stored event, sanitized, as the analyst sees it."""
    return {str(key): sanitize_text(value, 600) for key, value in row.items() if key != "raw" and not pd.isna(value)}


def resolve_alert(
    alert: dict[str, Any], events: pd.DataFrame, feed: DetectionFeed | None = None
) -> tuple[dict[str, Any], str]:
    """Swap a client's copy of a known event or detection for the stored record.

    The feeds show only a few fields per event, so a client posting an alert back
    may have dropped the very field that carries an injection payload. The stored
    record is the evidence; a client cannot trim or rewrite it. For detections the
    stored record is rebuilt with :func:`flow_alert`, which never includes the
    ground-truth label a client may have copied from ``/detections``.
    """
    event_id = alert.get("event_id")
    if not isinstance(event_id, str):
        return alert, "request"
    if "event_id" in events:
        match = events[events["event_id"] == event_id]
        if len(match):
            return alert_from_row(match.iloc[0]), "stored_event"
    if feed is not None:
        match = feed.detections[feed.detections["event_id"] == event_id]
        if len(match):
            return flow_alert(match.iloc[0]), "stored_detection"
    return alert, "request"


def run_investigation(
    alert: dict[str, Any], events: pd.DataFrame, retriever: HybridRetriever | None, llm: LLMClient | None
) -> dict[str, Any]:
    """Investigate one alert and package verdict, trace and report for a client."""
    analyst = llm or RuleBasedAnalyst()
    investigation = investigate(alert, analyst, ToolBox(events, retriever), retriever=retriever)
    return {
        "analyst": type(analyst).__name__,
        "verdict": investigation.verdict.model_dump(),
        "trace": investigation.trace,
        "usage": {"input_tokens": investigation.input_tokens, "output_tokens": investigation.output_tokens},
        "report_markdown": render_report(alert, investigation, retriever=retriever),
    }


def report_filename(alert: dict[str, Any], extension: str) -> str:
    """A download name built only from safe characters of the event id."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(alert.get("event_id", "alert")))[:80].strip("._") or "alert"
    return f"incident-{stem}.{extension}"


def create_app(
    events: pd.DataFrame,
    retriever: HybridRetriever | None = None,
    llm: LLMClient | None = None,
    *,
    feed: DetectionFeed | None = None,
    unavailable: dict[str, str] | None = None,
) -> FastAPI:
    """Build the API around host events, optional flow detections and knowledge base, and an analyst."""
    app = FastAPI(title="SentinelAI", version="0.1.0")
    tool_events = Resources(events, retriever, feed).investigation_events()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "events": len(events),
            "knowledge_base": retriever is not None,
            "detector": feed.summary if feed is not None else None,
            "unavailable": unavailable or {},
        }

    @app.get("/alerts")
    def alerts(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
        """Most recent events, sanitized, each flagged if it looks like a prompt injection."""
        recent = events.sort_values("timestamp", ascending=False).head(limit).astype(object)
        out: list[dict[str, Any]] = []
        for row in recent.to_dict(orient="records"):
            item: dict[str, Any] = {
                k: sanitize_text(row[k], 300) for k in _ALERT_FIELDS if k in row and not pd.isna(row[k])
            }
            text = " ".join(str(value) for value in row.values() if isinstance(value, str))
            item["injection_suspected"] = detect_injection(text).is_suspicious
            out.append(item)
        return out

    @app.get("/detections")
    def detections(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        """Most recent flagged flows.

        ``ground_truth`` (the dataset label) and ``source_event_id`` (the dataset's id,
        which names the capture file) are for people; the analyst never sees either.
        """
        if feed is None:
            raise HTTPException(status_code=503, detail="flow detector not loaded")
        recent = feed.detections.iloc[::-1].head(limit)
        alerts = [
            flow_alert(row) | {key: None if pd.isna(row.get(key)) else str(row[key]) for key in PEOPLE_ONLY}
            for _, row in recent.iterrows()
        ]
        return {"summary": feed.summary, "alerts": alerts}

    @app.get("/kb/search")
    def kb_search(q: str = Query(min_length=2, max_length=300), top_k: int = Query(default=5, ge=1, le=20)) -> Any:
        if retriever is None:
            raise HTTPException(status_code=503, detail="knowledge base not built")
        return [
            {"id": r.document.doc_id, "title": r.document.title, "kind": r.document.kind, "url": r.document.url}
            for r in retriever.search(q, top_k=top_k)
        ]

    @app.post("/investigate")
    def investigate_alert(request: InvestigateRequest) -> dict[str, Any]:
        alert, source = resolve_alert(request.alert, events, feed)
        return run_investigation(alert, tool_events, retriever, llm) | {"alert_source": source}

    @app.post("/reports", response_class=PlainTextResponse)
    def report(request: ReportRequest, format: Literal["markdown", "pdf"] = "markdown") -> Response:
        """Render an alert and verdict as a Markdown (default) or PDF report."""
        investigation = Investigation(verdict=request.verdict)
        if format == "pdf":
            pdf = render_report_pdf(request.alert, investigation, retriever=retriever)
            disposition = f'attachment; filename="{report_filename(request.alert, "pdf")}"'
            return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": disposition})
        markdown = render_report(request.alert, investigation, retriever=retriever)
        return PlainTextResponse(markdown, media_type="text/markdown")

    return app


def default_app() -> FastAPI:
    """App over the Sysmon sample plus whatever is built (``uvicorn --factory api.main:default_app``)."""
    resources = load_resources()
    return create_app(resources.events, resources.retriever, feed=resources.feed, unavailable=resources.unavailable)
