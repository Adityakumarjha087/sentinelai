"""Text embedders behind one interface."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Turns texts into L2-normalized vectors."""

    def embed(self, texts: Sequence[str]) -> Vectors: ...


class SentenceTransformerEmbedder:
    """Dense embeddings from a sentence-transformers model (downloaded on first use)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> Vectors:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. No download; for tests and offline use."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> Vectors:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                vectors[row, int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized: Vectors = (vectors / np.where(norms > 0, norms, 1.0)).astype(np.float32)
        return normalized
