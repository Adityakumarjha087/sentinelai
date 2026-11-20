"""Build and query the knowledge base (``sentinel-kb``)."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import httpx

from common.settings import get_settings
from data.download import download_file
from rag.documents import KBDocument, chunk_document, parse_attack, parse_nvd
from rag.embeddings import SentenceTransformerEmbedder
from rag.retriever import CrossEncoderReranker, HybridRetriever

logger = logging.getLogger("rag.cli")

ATTACK_FILE = "enterprise-attack-20.1.json"
# Pinned to a commit of mitre-attack/attack-stix-data and verified by checksum.
ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "f59b03ecf828579d5f9e93683ce25b7e2e5fd9d7/enterprise-attack/" + ATTACK_FILE
)
ATTACK_SHA256 = "539a843a7b53ed9749740b2123dddae3726b1a4831841c32aca740b011b910b5"
# CVE subset: those in CISA's Known Exploited Vulnerabilities catalog. This is a
# live API, so the snapshot is dated rather than checksummed.
NVD_KEV_FILE = "nvd_kev.json"
NVD_KEV_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0?hasKev&resultsPerPage=2000"


def fetch_sources(raw_dir: Path) -> tuple[Path, Path]:
    """Download the ATT&CK bundle and the known-exploited CVE snapshot if missing."""
    attack = download_file(ATTACK_URL, raw_dir / ATTACK_FILE, expected_sha256=ATTACK_SHA256)
    nvd = raw_dir / NVD_KEV_FILE
    if not nvd.exists():
        response = httpx.get(NVD_KEV_URL, timeout=120.0, follow_redirects=True)
        response.raise_for_status()
        nvd.write_bytes(response.content)
    return attack, nvd


def load_documents(attack_path: Path, nvd_path: Path) -> list[KBDocument]:
    """Parse both sources into documents."""
    attack = parse_attack(json.loads(attack_path.read_text(encoding="utf-8")))
    cves = parse_nvd(json.loads(nvd_path.read_text(encoding="utf-8")))
    logger.info("parsed %d ATT&CK techniques and %d CVEs", len(attack), len(cves))
    return [*attack, *cves]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns a process exit code."""
    parser = argparse.ArgumentParser(prog="sentinel-kb", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="download sources and build the index")
    search = commands.add_parser("search", help="query the index")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--mode", choices=("hybrid", "bm25", "vector"), default="hybrid")
    search.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    index_dir = settings.knowledge_base_dir
    if args.command == "build":
        documents = load_documents(*fetch_sources(settings.raw_dir / "kb"))
        chunks = [chunk for document in documents for chunk in chunk_document(document)]
        HybridRetriever.build(documents, chunks, SentenceTransformerEmbedder(), index_dir)
        logger.info("indexed %d chunks from %d documents", len(chunks), len(documents))
    else:
        reranker = None if args.no_rerank else CrossEncoderReranker()
        retriever = HybridRetriever.load(index_dir, SentenceTransformerEmbedder(), reranker)
        for result in retriever.search(args.query, args.top_k, mode=args.mode):
            print(f"{result.score:8.3f}  {result.document.doc_id:<16} {result.document.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
