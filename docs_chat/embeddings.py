"""Embedding backends.

* ``E5Embedder`` – the real one: ``intfloat/multilingual-e5-small`` via
  sentence-transformers. E5 models expect ``"query: "`` / ``"passage: "`` prefixes;
  vectors are L2-normalised so inner product == cosine similarity.
* ``HashEmbedder`` – a deterministic bag-of-hashed-words embedder that needs no model
  download. Used by the test-suite and by ``EMBEDDER=hash`` for offline smoke tests.
  It is *not* semantic; do not use it for real answers.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    name: str
    dim: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic, model-free embedder for tests and offline runs."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class E5Embedder:
    """multilingual-e5 via sentence-transformers (lazy-loaded)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", batch_size: int = 32) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self._model = None
        self.dim = 0

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, on demand

            self._model = SentenceTransformer(self.name, device="cpu")
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")


def get_embedder(kind: str = "e5", model_name: str = "intfloat/multilingual-e5-small") -> Embedder:
    if kind == "e5":
        return E5Embedder(model_name)
    if kind == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedder: {kind!r} (use 'e5' or 'hash')")
