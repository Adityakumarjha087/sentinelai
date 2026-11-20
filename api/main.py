"""Minimal read-only API over the event store, knowledge base and report renderer.

There is no ``/investigate`` endpoint yet: it needs a configured LLM provider.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict

from agent.investigator import Investigation
from agent.report import render_report
from agent.verdict import Verdict
from guardrails.injection import detect_injection
from guardrails.sanitize import sanitize_text
from rag.retriever import HybridRetriever

_ALERT_FIELDS = ("event_id", "timestamp", "event_type", "host", "process", "user", "src_ip", "dst_ip", "dst_port")


class ReportRequest(BaseModel):
    """An alert and the verdict reached for it."""

    model_config = ConfigDict(extra="forbid")

    alert: dict[str, Any]
    verdict: Verdict


def create_app(events: pd.DataFrame, retriever: HybridRetriever | None = None) -> FastAPI:
    """Build the API around an event table and an optional knowledge base."""
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

    @app.post("/reports", response_class=PlainTextResponse)
    def report(request: ReportRequest) -> PlainTextResponse:
        markdown = render_report(request.alert, Investigation(verdict=request.verdict), retriever=retriever)
        return PlainTextResponse(markdown, media_type="text/markdown")

    return app
