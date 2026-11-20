"""Hybrid retrieval: BM25 + dense vectors, fused with RRF, optionally reranked.

* BM25 finds exact identifiers and rare terms (``T1059.001``, ``lsass.exe``).
* Dense vectors (ChromaDB) find paraphrases ("credential theft" ~ "dump passwords").
* Reciprocal rank fusion merges the two rankings without tuning score scales.
* A cross-encoder reranker, when supplied, re-scores the fused shortlist.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Okapi

from rag.documents import Chunk, KBDocument, Kind
from rag.embeddings import Embedder

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_COLLECTION = "sentinel_kb"
_DOCUMENTS_FILE = "documents.jsonl"
_CHUNKS_FILE = "chunks.jsonl"
# Keeps security identifiers whole: T1059.001, CVE-2021-44228, lsass.exe, 10.0.0.1
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")
_RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; compound identifiers are kept whole *and* split into parts."""
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        tokens.append(token)
        parts = re.split(r"[.\-_]", token)
        if len(parts) > 1:
            tokens.extend(part for part in parts if part)
    return tokens


class Reranker(Protocol):
    """Scores (query, passage) pairs; higher is more relevant."""

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...


class CrossEncoderReranker:
    """Cross-encoder reranker (downloaded on first use)."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        scores = self._model.predict([(query, passage) for passage in passages])
        return [float(score) for score in scores]


class SearchResult(BaseModel):
    """A retrieved document with its best-matching passage."""

    model_config = ConfigDict(frozen=True)

    document: KBDocument
    passage: str
    score: float


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = _RRF_K) -> dict[str, float]:
    """Fuse ranked id lists: each list contributes ``1 / (k + rank)`` per id."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """Searchable knowledge base persisted in a directory."""

    def __init__(
        self,
        documents: Sequence[KBDocument],
        chunks: Sequence[Chunk],
        embedder: Embedder,
        collection: chromadb.Collection,
        reranker: Reranker | None = None,
    ) -> None:
        self._documents = {document.doc_id: document for document in documents}
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
        self._embedder = embedder
        self._collection = collection
        self.reranker = reranker

    @staticmethod
    def _client(directory: Path) -> ClientAPI:
        return chromadb.PersistentClient(
            path=str(directory / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
        )

    @classmethod
    def build(
        cls,
        documents: Sequence[KBDocument],
        chunks: Sequence[Chunk],
        embedder: Embedder,
        directory: Path,
        reranker: Reranker | None = None,
    ) -> HybridRetriever:
        """Embed every chunk and persist the index into ``directory`` (replacing any old one)."""
        if not chunks:
            raise ValueError("cannot build a knowledge base with no chunks")
        directory.mkdir(parents=True, exist_ok=True)
        client = cls._client(directory)
        if _COLLECTION in {collection.name for collection in client.list_collections()}:
            client.delete_collection(_COLLECTION)
        collection = client.create_collection(_COLLECTION, metadata={"hnsw:space": "cosine"})
        for start in range(0, len(chunks), 512):
            batch = chunks[start : start + 512]
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embedder.embed([chunk.text for chunk in batch]),
                metadatas=[{"kind": chunk.kind, "doc_id": chunk.doc_id} for chunk in batch],
            )
        for name, items in ((_DOCUMENTS_FILE, documents), (_CHUNKS_FILE, chunks)):
            lines = "\n".join(item.model_dump_json() for item in items)
            (directory / name).write_text(lines + "\n", encoding="utf-8")
        return cls(documents, chunks, embedder, collection, reranker)

    @classmethod
    def load(
        cls, directory: Path, embedder: Embedder, reranker: Reranker | None = None
    ) -> HybridRetriever:
        """Open an index written by :meth:`build`. Use the same embedder it was built with."""
        documents = [
            KBDocument(**json.loads(line))
            for line in (directory / _DOCUMENTS_FILE).read_text(encoding="utf-8").splitlines()
        ]
        chunks = [
            Chunk(**json.loads(line))
            for line in (directory / _CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
        ]
        collection = cls._client(directory).get_collection(_COLLECTION)
        return cls(documents, chunks, embedder, collection, reranker)

    def get(self, doc_id: str) -> KBDocument | None:
        """Look a document up by exact id (case-insensitive)."""
        return self._documents.get(doc_id.upper())

    def _keyword(self, query: str, kind: Kind | None, limit: int) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = [self._chunk_ids[i] for i in np.argsort(-scores) if scores[i] > 0]
        if kind is not None:
            ranked = [chunk_id for chunk_id in ranked if self._chunks[chunk_id].kind == kind]
        return ranked[:limit]

    def _vector(self, query: str, kind: Kind | None, limit: int) -> list[str]:
        result = self._collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=min(limit, len(self._chunk_ids)),
            where={"kind": kind} if kind is not None else None,
        )
        return [str(chunk_id) for chunk_id in result["ids"][0]]

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        kind: Kind | None = None,
        mode: str = "hybrid",
        candidates: int = 30,
    ) -> list[SearchResult]:
        """Return the ``top_k`` most relevant documents for ``query``.

        ``mode`` is ``"hybrid"`` (default), ``"bm25"`` or ``"vector"``. Results are
        de-duplicated per document, keeping each document's best passage.
        """
        if not query.strip():
            return []
        if mode not in {"hybrid", "bm25", "vector"}:
            raise ValueError(f"unknown search mode: {mode}")
        rankings: list[list[str]] = []
        if mode in {"hybrid", "bm25"}:
            rankings.append(self._keyword(query, kind, candidates))
        if mode in {"hybrid", "vector"}:
            rankings.append(self._vector(query, kind, candidates))
        fused = reciprocal_rank_fusion(rankings)
        shortlist = sorted(fused, key=lambda chunk_id: -fused[chunk_id])[:candidates]
        if self.reranker is not None and shortlist:
            scores = self.reranker.score(query, [self._chunks[c].text for c in shortlist])
            fused = dict(zip(shortlist, scores, strict=True))
            shortlist = sorted(shortlist, key=lambda chunk_id: -fused[chunk_id])

        results: list[SearchResult] = []
        seen: set[str] = set()
        for chunk_id in shortlist:
            chunk = self._chunks[chunk_id]
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            results.append(
                SearchResult(
                    document=self._documents[chunk.doc_id],
                    passage=chunk.text,
                    score=float(fused[chunk_id]),
                )
            )
            if len(results) == top_k:
                break
        return results
