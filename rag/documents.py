"""Parse MITRE ATT&CK (STIX 2.1) and NVD CVE JSON into documents, and chunk them.

Knowledge-base text is public reference data, but it still reaches the LLM, so
later phases treat retrieved passages as data, not instructions.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Kind = Literal["attack", "cve"]
_CITATION = re.compile(r"\(Citation:[^)]*\)")
_MARKUP = re.compile(r"</?code>|\[([^\]]+)\]\([^)]+\)")


class KBDocument(BaseModel):
    """One ATT&CK technique or one CVE."""

    model_config = ConfigDict(frozen=True)

    doc_id: str  # "T1059.001" or "CVE-2021-44228"
    kind: Kind
    title: str
    text: str
    url: str
    tactics: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    mitigations: tuple[str, ...] = ()
    severity: str | None = None
    cvss: float | None = None


class Chunk(BaseModel):
    """A retrievable passage of a document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    kind: Kind
    title: str
    text: str


def _clean(text: str) -> str:
    text = _CITATION.sub("", text)
    text = _MARKUP.sub(lambda match: match.group(1) or "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _attack_id(obj: dict[str, Any]) -> tuple[str, str] | None:
    for reference in obj.get("external_references", []):
        if reference.get("source_name") == "mitre-attack" and "external_id" in reference:
            return str(reference["external_id"]), str(reference.get("url", ""))
    return None


def parse_attack(bundle: dict[str, Any]) -> list[KBDocument]:
    """Extract active techniques (with their mitigations) from an ATT&CK STIX bundle."""
    objects = [obj for obj in bundle.get("objects", []) if not obj.get("revoked") and not obj.get("x_mitre_deprecated")]
    by_id = {obj["id"]: obj for obj in objects}
    mitigations: dict[str, list[str]] = {}
    for obj in objects:
        if obj["type"] == "relationship" and obj.get("relationship_type") == "mitigates":
            source = by_id.get(obj["source_ref"])
            if source is not None and source["type"] == "course-of-action":
                mitigations.setdefault(obj["target_ref"], []).append(str(source["name"]))

    documents: list[KBDocument] = []
    for obj in objects:
        reference = _attack_id(obj) if obj["type"] == "attack-pattern" else None
        if reference is None:
            continue
        documents.append(
            KBDocument(
                doc_id=reference[0],
                kind="attack",
                title=str(obj["name"]),
                text=_clean(str(obj.get("description", ""))),
                url=reference[1],
                tactics=tuple(
                    str(phase["phase_name"])
                    for phase in obj.get("kill_chain_phases", [])
                    if phase.get("kill_chain_name") == "mitre-attack"
                ),
                platforms=tuple(str(p) for p in obj.get("x_mitre_platforms", [])),
                mitigations=tuple(sorted(set(mitigations.get(obj["id"], [])))),
            )
        )
    return sorted(documents, key=lambda document: document.doc_id)


def _cvss(metrics: dict[str, Any]) -> tuple[float | None, str | None]:
    for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key) or []
        if entries:
            data = entries[0].get("cvssData", {})
            severity = data.get("baseSeverity") or entries[0].get("baseSeverity")
            score = data.get("baseScore")
            return (float(score) if score is not None else None), severity
    return None, None


def parse_nvd(payload: dict[str, Any]) -> list[KBDocument]:
    """Extract CVEs from an NVD API 2.0 response."""
    documents: list[KBDocument] = []
    for item in payload.get("vulnerabilities", []):
        cve = item.get("cve", {})
        description = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), "")
        if not cve.get("id") or not description:
            continue
        score, severity = _cvss(cve.get("metrics", {}))
        action = cve.get("cisaRequiredAction")
        text = _clean(description) + (f"\n\nRequired action: {action}" if action else "")
        documents.append(
            KBDocument(
                doc_id=str(cve["id"]),
                kind="cve",
                title=str(cve.get("cisaVulnerabilityName") or cve["id"]),
                text=text,
                url=f"https://nvd.nist.gov/vuln/detail/{cve['id']}",
                severity=severity,
                cvss=score,
            )
        )
    return sorted(documents, key=lambda document: document.doc_id)


def chunk_document(document: KBDocument, max_words: int = 160, overlap: int = 30) -> list[Chunk]:
    """Split a document into overlapping word windows, each prefixed with its identity.

    The prefix (id, title, tactics) keeps every passage self-describing, so a
    chunk from the middle of a long technique still matches queries by name.
    """
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must be >= 0 and smaller than max_words")
    context = f" ({', '.join(document.tactics)})" if document.tactics else ""
    prefix = f"{document.doc_id} {document.title}{context}: "
    words = document.text.split()
    step = max_words - overlap
    starts = range(0, max(len(words) - overlap, 1), step)
    return [
        Chunk(
            chunk_id=f"{document.doc_id}#{index}",
            doc_id=document.doc_id,
            kind=document.kind,
            title=document.title,
            text=prefix + " ".join(words[start : start + max_words]),
        )
        for index, start in enumerate(starts)
    ]
