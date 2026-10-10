"""Milestone 2 — query → top-k chunks, with the weak-retrieval check.

Contract (implement in M2, keep the signature so app.py and eval.py can rely on it):

    retrieve(index, query, embedder, settings) -> RetrievalResult

where RetrievalResult has:
    hits: list[Hit]            # from DocIndex.hybrid_search, top_k long
    weak: bool                 # True if best dense score < settings.weak_retrieval_threshold
    best_dense_score: float | None

Optional reranker (behind a flag, off by default): rerank the fused top-``candidates``
with ``BAAI/bge-reranker-v2-m3`` and keep top_k. Too slow for the free CPU demo; used
in the eval to measure the gain.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .embeddings import Embedder
from .index import DocIndex, Hit


@dataclass
class RetrievalResult:
    hits: list[Hit]
    weak: bool
    best_dense_score: float | None


RETRIEVAL_MODES = ("hybrid", "dense")


def retrieve(
    index: DocIndex, query: str, embedder: Embedder, settings: Settings, mode: str = "hybrid"
) -> RetrievalResult:
    """Top-k chunks (hybrid RRF by default, or dense only for the eval) + the weak-retrieval flag."""
    if mode == "hybrid":
        hits = index.hybrid_search(
            query, embedder, k=settings.top_k, candidates=settings.candidates, rrf_k=settings.rrf_k
        )
    elif mode == "dense":
        hits = index.dense_only(query, embedder, k=settings.top_k)
    else:
        raise ValueError(f"Unknown retrieval mode {mode!r}; expected one of {RETRIEVAL_MODES}")
    dense_scores = [h.dense_score for h in hits if h.dense_score is not None]
    best = max(dense_scores) if dense_scores else None
    weak = best is None or best < settings.weak_retrieval_threshold
    return RetrievalResult(hits=hits, weak=weak, best_dense_score=best)
