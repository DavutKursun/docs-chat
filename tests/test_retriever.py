"""The weak-retrieval rule: BM25-based by default, dense threshold optional, cross-lingual queries never flagged."""

from __future__ import annotations

from docs_chat.chunking import Chunk
from docs_chat.config import Settings
from docs_chat.embeddings import HashEmbedder
from docs_chat.index import DocIndex
from docs_chat.retriever import retrieve

TEXTS = [
    "The 3-stage filter must be replaced every six months. Replacement filters carry the part number AF-200.",
    "Error code E3 means the device has overheated: unplug it and let it cool for thirty minutes.",
    "Full-time employees receive twenty-four working days of paid annual leave per calendar year.",
    "Salaries are paid on the 25th of each month by bank transfer; payslips are in the HR portal.",
    "The warranty covers defects in materials and workmanship for twenty-four months.",
]


def make_index(embedder):
    chunks = [Chunk(chunk_id=f"d:{i}:0", doc_name="d.pdf", page_no=i + 1, text=t, section_hint=t[:20]) for i, t in enumerate(TEXTS)]
    return DocIndex.build(chunks, embedder)


def settings(**over) -> Settings:
    return Settings(**{"embedder": "hash", "top_k": 3, "weak_retrieval_threshold": 0.0, "weak_bm25_threshold": 15.0, **over})


def test_bm25_rule_flags_weak_lexical_overlap_but_not_zero_overlap(embedder):
    index = make_index(embedder)
    strong = retrieve(index, "What does error code E3 mean?", embedder, settings())
    assert strong.best_bm25_score > 0 and strong.best_dense_score is not None
    # the same query with a threshold above its score is weak; a threshold of 0 switches the rule off
    assert retrieve(index, "What does error code E3 mean?", embedder, settings(weak_bm25_threshold=1e9)).weak is True
    assert retrieve(index, "What does error code E3 mean?", embedder, settings(weak_bm25_threshold=0)).weak is False
    # no word in common with the corpus (a Turkish question): BM25 = 0, never weak, even with a huge threshold
    turkish = retrieve(index, "Yeni çalışanlar için deneme süresi ne kadar?", embedder, settings(weak_bm25_threshold=1e9))
    assert turkish.best_bm25_score == 0.0 and turkish.weak is False
    assert len(turkish.hits) == 3  # dense retrieval still returns passages


def test_default_thresholds_leave_strong_matches_unflagged(embedder):
    index = make_index(embedder)
    result = retrieve(index, "What does error code E3 mean?", embedder, settings())
    assert result.weak == (0 < result.best_bm25_score < 15)


def test_dense_threshold_is_optional_and_independent(embedder):
    index = make_index(embedder)
    query = "What does error code E3 mean?"
    assert retrieve(index, query, embedder, settings(weak_retrieval_threshold=0.0, weak_bm25_threshold=0.0)).weak is False
    # hash cosine scores are small, so a high dense threshold flags the query even though BM25 is strong
    assert retrieve(index, query, embedder, settings(weak_retrieval_threshold=0.99, weak_bm25_threshold=0.0)).weak is True


def test_dense_mode_also_computes_the_bm25_signal(embedder):
    index = make_index(embedder)
    result = retrieve(index, "part number of the replacement filter AF-200", embedder, settings(), mode="dense")
    assert all(h.bm25_score is None for h in result.hits) and result.best_bm25_score > 0


def test_empty_index_is_not_weak_by_default(embedder):
    index = DocIndex.build([], HashEmbedder(dim=embedder.dim))
    result = retrieve(index, "anything", embedder, settings())
    assert result.hits == [] and result.weak is False and result.best_dense_score is None and result.best_bm25_score == 0.0
