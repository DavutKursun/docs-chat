"""Dense (FAISS) + keyword (BM25) index over chunks, with hybrid search.

Hybrid search = reciprocal rank fusion (RRF) of the dense top-N and the BM25 top-N.
The index is saved as three files in a directory: ``faiss.index``, ``chunks.json``,
``meta.json``; BM25 is rebuilt from the chunk texts on load (fast, no extra file).
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _import_faiss():
    """Import faiss so that it can share a process with torch.

    faiss-cpu and torch each bundle their own OpenMP runtime (libomp). On macOS two things
    go wrong when both are loaded: if faiss's copy is loaded first, torch segfaults later
    (ask.py loads the index before the e5 embedder), so torch is imported first when it is
    installed; and LLVM's libomp aborts at the first faiss search ("OMP: Error #15 ...
    already initialized") unless KMP_DUPLICATE_LIB_OK is set, the same workaround
    sentence-transformers applies on import. Both are no-ops without torch (hash embedder, CI).
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import faiss

    return faiss


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; Unicode-aware so Turkish characters survive."""
    return _TOKEN.findall(text.lower())


def rrf_fuse(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion.

    Each list is a ranking of item ids (best first). Score(id) = Σ 1 / (k + rank),
    rank starting at 1. Returns (id, score) sorted by score descending.
    """
    scores: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused (RRF) score
    dense_score: float | None  # cosine similarity if the chunk was in the dense list
    bm25_score: float | None


class DocIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        faiss = _import_faiss()
        from rank_bm25 import BM25Plus

        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.embedder_name = embedder_name
        self.dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
        self._faiss = faiss.IndexFlatIP(self.dim)
        if len(chunks):
            self._faiss.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._bm25 = BM25Plus([tokenize(c.text) for c in chunks]) if chunks else None

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        batch_size: int = 64,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> "DocIndex":
        """Embed the chunks in batches; ``progress_cb(done, total)`` is called after each batch."""
        if not chunks:
            return cls([], np.zeros((0, max(embedder.dim, 1)), dtype=np.float32), embedder.name)
        parts = []
        for i in range(0, len(chunks), batch_size):
            parts.append(embedder.encode_passages([c.text for c in chunks[i : i + batch_size]]))
            if progress_cb is not None:
                progress_cb(min(i + batch_size, len(chunks)), len(chunks))
        vectors = np.vstack(parts).astype(np.float32)
        return cls(chunks, vectors, embedder.name)

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        faiss = _import_faiss()

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss, str(directory / "faiss.index"))
        (directory / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            "embedder": self.embedder_name,
            "dim": self.dim,
            "n_chunks": len(self.chunks),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "DocIndex":
        faiss = _import_faiss()

        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))]
        faiss_index = faiss.read_index(str(directory / "faiss.index"))
        vectors = faiss_index.reconstruct_n(0, faiss_index.ntotal) if faiss_index.ntotal else np.zeros((0, meta["dim"]), dtype=np.float32)
        index = cls(chunks, np.asarray(vectors, dtype=np.float32), meta["embedder"])
        return index

    # ------------------------------------------------------------------ search
    def dense_search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, ids = self._faiss.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.chunks or self._bm25 is None:
            return []
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        order = np.argsort(-scores, kind="stable")[: min(k, len(self.chunks))]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def hybrid_search(
        self,
        query: str,
        embedder: Embedder,
        k: int = 6,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> list[Hit]:
        """Top-k chunks by RRF of dense and BM25 rankings."""
        if not self.chunks:
            return []
        qv = embedder.encode_queries([query])[0]
        dense = self.dense_search(qv, candidates)
        sparse = self.bm25_search(query, candidates)
        dense_scores = dict(dense)
        sparse_scores = dict(sparse)
        fused = rrf_fuse([[i for i, _ in dense], [i for i, _ in sparse]], k=rrf_k)
        hits: list[Hit] = []
        for idx, score in fused[:k]:
            hits.append(Hit(self.chunks[idx], score, dense_scores.get(idx), sparse_scores.get(idx)))
        return hits

    # ------------------------------------------------------------------- stats
    @property
    def documents(self) -> dict[str, int]:
        """doc_name -> number of pages that produced chunks."""
        pages: dict[str, set[int]] = {}
        for c in self.chunks:
            pages.setdefault(c.doc_name, set()).add(c.page_no)
        return {name: len(p) for name, p in pages.items()}

    def __len__(self) -> int:
        return len(self.chunks)
