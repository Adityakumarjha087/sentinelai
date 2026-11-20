"""Minimal read-only API over the event store, knowledge base and report renderer.

``/investigate`` uses whichever analyst the app was created with. Without a
configured LLM that is the offline rule-based baseline, and responses say so.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict

from agent.investigator import Investigation, investigate
from agent.llm import LLMClient
from agent.report import render_report
from agent.tools import ToolBox
from agent.verdict import Verdict
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
        "report_markdown": render_report(alert, investigation, retriever=retriever),
    }


def create_app(events: pd.DataFrame, retriever: HybridRetriever | None = None, llm: LLMClient | None = None) -> FastAPI:
    """Build the API around an event table, an optional knowledge base and an analyst."""
    app = FastAPI(title="SentinelAI", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "events": len(events), "knowledge_base": retriever is not None}

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
        return run_investigation(request.alert, events, retriever, llm)

    @app.post("/reports", response_class=PlainTextResponse)
    def report(request: ReportRequest) -> PlainTextResponse:
        markdown = render_report(request.alert, Investigation(verdict=request.verdict), retriever=retriever)
        return PlainTextResponse(markdown, media_type="text/markdown")

    return app


def default_app() -> FastAPI:
    """App over the bundled synthetic Sysmon sample (``uvicorn --factory api.main:default_app``)."""
    from common.settings import get_settings
    from data.parsers.sysmon import parse_sysmon_file
    from data.schema import events_to_frame

    samples = sorted(get_settings().samples_dir.glob("*.xml"))
    return create_app(events_to_frame([event for path in samples for event in parse_sysmon_file(path)]))
