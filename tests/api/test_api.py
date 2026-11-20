from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from data.parsers.sysmon import parse_sysmon_file
from data.schema import events_to_frame
from rag.documents import KBDocument, chunk_document
from rag.embeddings import HashingEmbedder
from rag.retriever import HybridRetriever


@pytest.fixture(scope="module")
def client(sysmon_sample: Path, tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    document = KBDocument(
        doc_id="T1046", kind="attack", title="Network Service Discovery", url="https://attack.mitre.org/T1046",
        text="Adversaries may scan ports to list services running on remote hosts.",
    )  # fmt: skip
    retriever = HybridRetriever.build(
        [document], chunk_document(document), HashingEmbedder(), tmp_path_factory.mktemp("kb")
    )
    return TestClient(create_app(events_to_frame(parse_sysmon_file(sysmon_sample)), retriever))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "events": 13, "knowledge_base": True}


def test_alerts_are_recent_first_and_flag_injection(client: TestClient) -> None:
    alerts = client.get("/alerts", params={"limit": 13}).json()
    assert len(alerts) == 13
    stamps = [alert["timestamp"] for alert in alerts]
    assert stamps == sorted(stamps, reverse=True)
    assert sum(alert["injection_suspected"] for alert in alerts) == 1
    assert len(client.get("/alerts", params={"limit": 2}).json()) == 2
    assert client.get("/alerts", params={"limit": 0}).status_code == 422


def test_kb_search(client: TestClient) -> None:
    results = client.get("/kb/search", params={"q": "scan ports"}).json()
    assert results[0]["id"] == "T1046"
    assert client.get("/kb/search", params={"q": "x"}).status_code == 422


def test_kb_search_without_a_knowledge_base(sysmon_sample: Path) -> None:
    bare = TestClient(create_app(events_to_frame(parse_sysmon_file(sysmon_sample))))
    assert bare.get("/kb/search", params={"q": "scan ports"}).status_code == 503
    assert bare.get("/health").json()["knowledge_base"] is False


def test_report_endpoint_renders_markdown_and_rejects_bad_verdicts(client: TestClient) -> None:
    verdict = {"is_malicious": True, "confidence": 0.8, "attack_techniques": ["T1046"], "evidence": ["many ports"]}
    response = client.post("/reports", json={"alert": {"event_id": "a-1"}, "verdict": verdict})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "# Incident report - a-1" in response.text
    assert "Network Service Discovery" in response.text
    bad = client.post("/reports", json={"alert": {}, "verdict": verdict | {"confidence": 9}})
    assert bad.status_code == 422
