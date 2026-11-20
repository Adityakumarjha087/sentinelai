"""Read-only investigation tools behind a strict allow-list.

Every tool only *reads*: there is no tool that blocks an IP, isolates a host or
deletes anything. Response steps are returned as recommendations that a human
must approve. Tool output containing log text is sanitized and delimited as
untrusted before it goes back to the model.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from collections.abc import Callable
from typing import Any

import pandas as pd

from guardrails.sanitize import sanitize_text, wrap_untrusted
from rag.retriever import HybridRetriever

_EVENT_COLUMNS = [  # noqa: E501
    "event_id",
    "timestamp",
    "event_type",
    "src_ip",
    "dst_ip",
    "dst_port",
    "protocol",
    "host",
    "process",
    "user",
]
MAX_ROWS = 20


class ToolNotAllowedError(Exception):
    """The model asked for a tool that is not on the allow-list."""


def _spec(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
    }  # fmt: skip


TOOL_SPECS: list[dict[str, Any]] = [
    _spec("query_logs", "Find events matching simple filters.", {
        "src_ip": {"type": "string"}, "dst_ip": {"type": "string"}, "host": {"type": "string"},
        "event_type": {"type": "string"}, "limit": {"type": "integer"}}, []),
    _spec("get_related_events", "Events sharing a host or IP with the given event, within a time window.", {
        "event_id": {"type": "string"}, "window_minutes": {"type": "integer"}}, ["event_id"]),
    _spec("get_ip_reputation", "Reputation of an IP address (mock threat-intel source).",
          {"ip": {"type": "string"}}, ["ip"]),
    _spec("search_attack_kb", "Search MITRE ATT&CK techniques and known-exploited CVEs.",
          {"query": {"type": "string"}}, ["query"]),
    _spec("score_severity", "Deterministic severity score from investigation facts.", {
        "is_malicious": {"type": "boolean"}, "privileged_access": {"type": "boolean"},
        "external_communication": {"type": "boolean"}, "hosts_affected": {"type": "integer"}},
        ["is_malicious"]),
]  # fmt: skip
ALLOWED_TOOLS = frozenset(spec["name"] for spec in TOOL_SPECS)


def score_severity(
    is_malicious: bool, privileged_access: bool = False, external_communication: bool = False, hosts_affected: int = 1
) -> dict[str, Any]:  # fmt: skip
    """Transparent, rule-based severity so the rating never depends on model mood."""
    if not is_malicious:
        return {"severity": "low", "score": 1}
    score = 4 + 3 * bool(privileged_access) + 2 * bool(external_communication) + min(max(hosts_affected, 1) - 1, 1)
    severity = "critical" if score >= 9 else "high" if score >= 6 else "medium"
    return {"severity": severity, "score": score}


def mock_ip_reputation(ip: str) -> dict[str, Any]:
    """Deterministic stand-in for a threat-intel lookup. Clearly labelled as mock data."""
    try:
        address = ipaddress.ip_address(ip.strip())
    except ValueError:
        return {"ip": sanitize_text(ip, 60), "error": "not a valid IP address"}
    if address.is_private or address.is_loopback:
        return {"ip": str(address), "scope": "internal", "abuse_score": 0, "source": "mock"}
    score = int(hashlib.sha256(str(address).encode()).hexdigest(), 16) % 101
    return {"ip": str(address), "scope": "external", "abuse_score": score, "source": "mock"}


class ToolBox:
    """Executes allow-listed tools over an event table and the knowledge base."""

    def __init__(self, events: pd.DataFrame, retriever: HybridRetriever | None = None) -> None:
        self._events = events
        self._retriever = retriever
        self._handlers: dict[str, Callable[..., Any]] = {
            "query_logs": self.query_logs,
            "get_related_events": self.get_related_events,
            "get_ip_reputation": mock_ip_reputation,
            "search_attack_kb": self.search_attack_kb,
            "score_severity": score_severity,
        }

    def _rows(self, frame: pd.DataFrame, limit: int) -> str:
        columns = [column for column in _EVENT_COLUMNS if column in frame.columns]
        if "command_line" in frame.columns:
            columns.append("command_line")
        rows = frame[columns].head(max(1, min(int(limit), MAX_ROWS))).astype(object)
        records = [
            {k: sanitize_text(v, 400) for k, v in row.items() if not pd.isna(v)}
            for row in rows.to_dict(orient="records")
        ]
        return wrap_untrusted(json.dumps({"matched": len(frame), "events": records}, default=str))

    def query_logs(self, limit: int = 10, **filters: str) -> str:
        frame = self._events
        for column, value in filters.items():
            if column not in {"src_ip", "dst_ip", "host", "event_type"}:
                raise ValueError(f"unsupported filter: {column}")
            frame = frame[frame[column] == value]
        return self._rows(frame, limit)

    def get_related_events(self, event_id: str, window_minutes: int = 10) -> str:
        match = self._events[self._events["event_id"] == event_id]
        if match.empty:
            return json.dumps({"error": "unknown event_id"})
        event = match.iloc[0]
        window = pd.Timedelta(minutes=max(1, min(int(window_minutes), 120)))
        near = self._events[(self._events["timestamp"] - event["timestamp"]).abs() <= window]
        related = pd.Series(False, index=near.index)
        for column in ("host", "src_ip", "dst_ip"):
            if not pd.isna(event[column]):
                related |= (near[column] == event[column]).fillna(False)
        return self._rows(near[related & (near["event_id"] != event_id)], MAX_ROWS)

    def search_attack_kb(self, query: str) -> str:
        if self._retriever is None:
            return json.dumps({"error": "knowledge base unavailable"})
        results = self._retriever.search(sanitize_text(query, 300), top_k=4)
        return json.dumps(
            [{"id": r.document.doc_id, "title": r.document.title, "tactics": r.document.tactics,
              "passage": r.passage[:500], "mitigations": r.document.mitigations[:5]} for r in results]
        )  # fmt: skip

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool. Unknown tools are refused; bad arguments return an error string."""
        if name not in ALLOWED_TOOLS:
            raise ToolNotAllowedError(name)
        try:
            result = self._handlers[name](**arguments)
        except (TypeError, ValueError, KeyError) as exc:
            return json.dumps({"error": f"invalid arguments: {type(exc).__name__}"})
        return result if isinstance(result, str) else json.dumps(result, default=str)
