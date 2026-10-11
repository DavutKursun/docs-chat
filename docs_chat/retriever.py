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
    best_bm25_score: float = 0.0  # top BM25 score in the whole index; 0 = no word of the query occurs in it


RETRIEVAL_MODES = ("hybrid", "dense")


def retrieve(
    index: DocIndex, query: str, embedder: Embedder, settings: Settings, mode: str = "hybrid"
) -> RetrievalResult:
    """Top-k chunks (hybrid RRF by default, or dense only for the eval) + the weak-retrieval flag.

    Weak retrieval (the LLM is told the answer may not be in the documents) is flagged when
    * ``settings.weak_bm25_threshold`` > 0 and 0 < best BM25 score < that threshold: the question shares
      only weak lexical evidence with the corpus. A best score of exactly 0 is *not* weak: the query has no
      word in common with the documents, which is the normal case for a Turkish question on English
      documents (cross-lingual retrieval is dense-only), so there is no lexical signal either way; or
    * ``settings.weak_retrieval_threshold`` > 0 and the best dense cosine is below it (off by default:
      e5-small scores of answerable and unanswerable questions overlap, see docs/DESIGN.md).
    """
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
    top_bm25 = index.bm25_search(query, 1)
    best_bm25 = top_bm25[0][1] if top_bm25 else 0.0
    weak_dense = settings.weak_retrieval_threshold > 0 and (best is None or best < settings.weak_retrieval_threshold)
    weak_bm25 = settings.weak_bm25_threshold > 0 and 0 < best_bm25 < settings.weak_bm25_threshold
    return RetrievalResult(hits=hits, weak=weak_dense or weak_bm25, best_dense_score=best, best_bm25_score=best_bm25)
