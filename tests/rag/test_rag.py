from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rag.documents import KBDocument, chunk_document, parse_attack, parse_nvd
from rag.embeddings import HashingEmbedder
from rag.retriever import HybridRetriever, _writable_chroma_dir, reciprocal_rank_fusion, tokenize


def technique(stix_id: str, ext_id: str, name: str, text: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "attack-pattern",
        "id": stix_id,
        "name": name,
        "description": text,
        "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "execution"}],
        "x_mitre_platforms": ["Windows"],
        "external_references": [
            {"source_name": "other", "external_id": "X1"},
            {
                "source_name": "mitre-attack",
                "external_id": ext_id,
                "url": f"https://attack.mitre.org/{ext_id}",
            },
        ],
        **extra,
    }


BUNDLE: dict[str, Any] = {
    "objects": [
        technique("ap--1", "T1059.001", "PowerShell",
                  "Adversaries may abuse <code>powershell.exe</code> commands and scripts for execution."
                  " (Citation: Some Report) See [PowerShell](https://example.com/ps)."),
        technique("ap--2", "T1003.001", "LSASS Memory",
                  "Adversaries may dump credentials stored in lsass.exe process memory to steal passwords."),
        technique("ap--3", "T1046", "Network Service Discovery",
                  "Adversaries may scan ports to list services running on remote hosts."),
        technique("ap--4", "T9999", "Old Technique", "Gone.", revoked=True),
        technique("ap--5", "T9998", "Deprecated Technique", "Gone.", x_mitre_deprecated=True),
        {"type": "course-of-action", "id": "coa--1", "name": "Code Signing"},
        {"type": "relationship", "id": "rel--1", "relationship_type": "mitigates",
         "source_ref": "coa--1", "target_ref": "ap--1"},
        {"type": "relationship", "id": "rel--2", "relationship_type": "uses",
         "source_ref": "coa--1", "target_ref": "ap--2"},
    ]
}  # fmt: skip
NVD: dict[str, Any] = {
    "vulnerabilities": [
        {"cve": {
            "id": "CVE-2021-44228",
            "descriptions": [{"lang": "es", "value": "no"},
                             {"lang": "en", "value": "Apache Log4j2 JNDI features allow remote code execution."}],
            "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 10.0, "baseSeverity": "CRITICAL"}}]},
            "cisaVulnerabilityName": "Apache Log4j2 Remote Code Execution Vulnerability",
            "cisaRequiredAction": "Apply updates per vendor instructions.",
        }},
        {"cve": {"id": "CVE-0000-0001", "descriptions": []}},
    ]
}  # fmt: skip


@pytest.fixture(scope="module")
def documents() -> list[KBDocument]:
    return [*parse_attack(BUNDLE), *parse_nvd(NVD)]


@pytest.fixture(scope="module")
def retriever(documents: list[KBDocument], tmp_path_factory: pytest.TempPathFactory) -> HybridRetriever:
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    return HybridRetriever.build(documents, chunks, HashingEmbedder(), tmp_path_factory.mktemp("kb"))


def test_attack_parsing_skips_revoked_and_cleans_markup() -> None:
    parsed = {document.doc_id: document for document in parse_attack(BUNDLE)}
    assert sorted(parsed) == ["T1003.001", "T1046", "T1059.001"]
    powershell = parsed["T1059.001"]
    assert powershell.tactics == ("execution",)
    assert powershell.platforms == ("Windows",)
    assert powershell.mitigations == ("Code Signing",)
    assert parsed["T1003.001"].mitigations == ()  # a "uses" relationship is not a mitigation
    assert "Citation" not in powershell.text
    assert "<code>" not in powershell.text
    assert "example.com" not in powershell.text
    assert "powershell.exe" in powershell.text


def test_nvd_parsing_takes_english_text_and_cvss() -> None:
    (cve,) = parse_nvd(NVD)  # the entry without a description is skipped
    assert cve.doc_id == "CVE-2021-44228"
    assert (cve.cvss, cve.severity) == (10.0, "CRITICAL")
    assert cve.title.startswith("Apache Log4j2")
    assert "Required action: Apply updates" in cve.text


def test_chunks_overlap_and_carry_their_identity() -> None:
    document = KBDocument(
        doc_id="T1", kind="attack", title="Long", url="u", tactics=("impact",),
        text=" ".join(f"w{i}" for i in range(100)),
    )  # fmt: skip
    chunks = chunk_document(document, max_words=40, overlap=10)
    assert [chunk.chunk_id for chunk in chunks] == ["T1#0", "T1#1", "T1#2"]
    assert all(chunk.text.startswith("T1 Long (impact): ") for chunk in chunks)
    assert chunks[0].text.split()[-1] == "w39"
    assert chunks[1].text.split()[3] == "w30"  # after the 3-word prefix; 10 words of overlap
    assert chunks[-1].text.split()[-1] == "w99"
    assert len(chunk_document(document.model_copy(update={"text": ""}))) == 1
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(document, max_words=10, overlap=10)


def test_tokenizer_keeps_security_identifiers() -> None:
    tokens = tokenize("Seen T1059.001 and CVE-2021-44228 via lsass.exe")
    assert {"t1059.001", "t1059", "cve-2021-44228", "44228", "lsass.exe", "lsass"} <= set(tokens)


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder()
    vectors = embedder.embed(["dump lsass memory", "dump lsass memory", ""])
    assert np.array_equal(vectors[0], vectors[1])
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
    assert not vectors[2].any()


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["a"] == pytest.approx(fused["b"])
    assert fused["a"] > fused["c"] > 0
    assert fused["c"] == pytest.approx(fused["d"])


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "vector"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("process dumping lsass.exe memory to steal passwords", "T1003.001"),
        ("scan ports on remote hosts", "T1046"),
        ("powershell.exe scripts used for execution", "T1059.001"),
        ("Log4j2 JNDI remote code execution", "CVE-2021-44228"),
    ],
)
def test_search_finds_the_right_document(retriever: HybridRetriever, mode: str, query: str, expected: str) -> None:
    results = retriever.search(query, top_k=2, mode=mode)
    assert results[0].document.doc_id == expected
    assert results[0].passage.startswith(expected)


def test_exact_identifier_lookup_and_kind_filter(retriever: HybridRetriever) -> None:
    assert retriever.search("what is T1046", mode="bm25")[0].document.doc_id == "T1046"
    assert retriever.get("cve-2021-44228") is not None
    assert retriever.get("T0000") is None
    only_cves = retriever.search("remote code execution", top_k=5, kind="cve")
    assert [result.document.kind for result in only_cves] == ["cve"]
    only_attack = retriever.search("remote code execution", top_k=5, kind="attack")
    assert only_attack
    assert {result.document.kind for result in only_attack} == {"attack"}


def test_results_are_unique_documents_and_bounded(retriever: HybridRetriever) -> None:
    results = retriever.search("adversaries", top_k=2)
    ids = [result.document.doc_id for result in results]
    assert len(ids) == len(set(ids)) == 2
    assert retriever.search("   ") == []
    with pytest.raises(ValueError, match="unknown search mode"):
        retriever.search("x", mode="magic")


def test_reranker_reorders_the_shortlist(retriever: HybridRetriever) -> None:
    class PreferPortScans:
        def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
            return [1.0 if passage.startswith("T1046") else 0.0 for passage in passages]

    retriever.reranker = PreferPortScans()
    try:
        results = retriever.search("adversaries may", top_k=3)
    finally:
        retriever.reranker = None
    assert results[0].document.doc_id == "T1046"
    assert results[0].score == 1.0


def test_index_survives_a_reload(documents: list[KBDocument], tmp_path: Path) -> None:
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    built = HybridRetriever.build(documents, chunks, HashingEmbedder(), tmp_path)
    loaded = HybridRetriever.load(tmp_path, HashingEmbedder())
    query = "dump credentials from lsass"
    assert [r.document.doc_id for r in loaded.search(query)] == [r.document.doc_id for r in built.search(query)]
    rebuilt = HybridRetriever.build(documents[:1], chunks[:1], HashingEmbedder(), tmp_path)
    assert len(rebuilt.search("adversaries", top_k=5)) == 1  # the old index was replaced
    with pytest.raises(ValueError, match="no chunks"):
        HybridRetriever.build(documents, [], HashingEmbedder(), tmp_path)


def test_a_read_only_index_is_opened_from_a_writable_copy(documents: list[KBDocument], tmp_path: Path) -> None:
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    HybridRetriever.build(documents, chunks, HashingEmbedder(), tmp_path)
    assert _writable_chroma_dir(tmp_path) == tmp_path
    paths = [tmp_path / "chroma", *(tmp_path / "chroma").rglob("*")]
    for path in reversed(paths):  # files before their directories
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    try:
        copy = _writable_chroma_dir(tmp_path)
        assert copy != tmp_path
        assert all(os.access(path, os.W_OK) for path in (copy / "chroma").rglob("*"))
        loaded = HybridRetriever.load(tmp_path, HashingEmbedder())
        assert loaded.search("dump credentials from lsass")
    finally:
        for path in paths:
            path.chmod(stat.S_IMODE(path.stat().st_mode) | 0o200)
