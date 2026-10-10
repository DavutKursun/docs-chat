from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from docs_chat.chunking import Chunk, chunk_pages
from docs_chat.config import Settings
from docs_chat.embeddings import HashEmbedder, get_embedder
from docs_chat.index import DocIndex, rrf_fuse, tokenize
from docs_chat.loaders import load_folder
from docs_chat.retriever import retrieve


def _chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"d:1:{i}", doc_name="d", page_no=1, text=t, section_hint=t[:20]) for i, t in enumerate(texts)
    ]


def test_rrf_fuse_math():
    fused = rrf_fuse([[1, 2, 3], [2, 1]], k=60)
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[3] == pytest.approx(1 / 63)
    assert [i for i, _ in fused][-1] == 3
    assert rrf_fuse([], k=60) == []


def test_rrf_item_in_both_lists_beats_item_in_one():
    fused = rrf_fuse([[7, 8], [9, 7]], k=60)
    assert fused[0][0] == 7


def test_tokenize_is_unicode_aware():
    assert tokenize("Çalışanlar 24 gün izin alır!") == ["çalışanlar", "24", "gün", "izin", "alır"]


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(dim=64)
    a = e.encode_passages(["vacation days", "vacation days"])
    assert np.allclose(a[0], a[1])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert e.encode_queries(["x"]).shape == (1, 64)


def test_get_embedder_unknown_kind():
    with pytest.raises(ValueError):
        get_embedder("word2vec")


def test_build_search_and_roundtrip(tmp_path: Path, embedder: HashEmbedder):
    chunks = _chunks(
        [
            "Employees receive twenty four days of annual leave per year.",
            "The warranty of the purifier lasts twenty four months.",
            "Salaries are paid on the twenty fifth of each month.",
        ]
    )
    index = DocIndex.build(chunks, embedder)
    assert len(index) == 3

    hits = index.hybrid_search("how many days of annual leave", embedder, k=2)
    assert hits[0].chunk.chunk_id == "d:1:0"
    assert hits[0].dense_score is not None and hits[0].bm25_score is not None

    index.save(tmp_path / "idx")
    assert {p.name for p in (tmp_path / "idx").iterdir()} == {"faiss.index", "chunks.json", "meta.json"}
    loaded = DocIndex.load(tmp_path / "idx")
    assert len(loaded) == 3 and loaded.embedder_name == embedder.name and loaded.dim == 128
    hits2 = loaded.hybrid_search("how many days of annual leave", embedder, k=2)
    assert [h.chunk.chunk_id for h in hits2] == [h.chunk.chunk_id for h in hits]


def test_empty_index(embedder: HashEmbedder):
    index = DocIndex.build([], embedder)
    assert len(index) == 0
    assert index.hybrid_search("anything", embedder) == []


def test_bm25_finds_exact_keywords(embedder: HashEmbedder):
    chunks = _chunks(["Error code E3 means the device overheated.", "Error code E1 means the fan is blocked."])
    index = DocIndex.build(chunks, embedder)
    assert index.bm25_search("E3", k=1)[0][0] == 0


def test_sample_docs_retrieval_hits_gold_pages(sample_docs_dir: Path, embedder: HashEmbedder):
    """Keyword-heavy questions must retrieve the gold page even with the toy embedder."""
    pages, _ = load_folder(sample_docs_dir)
    index = DocIndex.build(chunk_pages(pages), embedder)
    settings = Settings()
    cases = [
        ("What is the part number of the replacement filter AF-200?", "aria_x200_manual.pdf"),
        ("How long is the probation period?", "lumen_employee_handbook.pdf"),
        ("What does error code E3 mean?", "aria_x200_manual.pdf"),
    ]
    for question, doc in cases:
        result = retrieve(index, question, embedder, settings)
        assert any(h.chunk.doc_name == doc for h in result.hits[:3]), question


def test_documents_summary(embedder: HashEmbedder):
    chunks = [
        Chunk(chunk_id="a:1:0", doc_name="a", page_no=1, text="x " * 30, section_hint="x"),
        Chunk(chunk_id="a:2:0", doc_name="a", page_no=2, text="y " * 30, section_hint="y"),
        Chunk(chunk_id="b:1:0", doc_name="b", page_no=1, text="z " * 30, section_hint="z"),
    ]
    assert DocIndex.build(chunks, embedder).documents == {"a": 2, "b": 1}


def test_import_faiss_works_with_and_without_torch(monkeypatch):
    import importlib.util
    import os
    import sys

    from docs_chat import index as index_module

    assert index_module._import_faiss().__name__ == "faiss"
    if importlib.util.find_spec("torch") is not None:  # the OpenMP workaround is only needed next to torch
        assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
    monkeypatch.setitem(sys.modules, "torch", None)  # simulates "torch is not installed"
    assert index_module._import_faiss().__name__ == "faiss"


def test_build_reports_embedding_progress_per_batch(embedder):
    chunks = [Chunk(chunk_id=f"d:1:{i}", doc_name="d", page_no=1, text=f"text {i}", section_hint="t") for i in range(5)]
    calls = []
    index = DocIndex.build(chunks, embedder, batch_size=2, progress_cb=lambda done, total: calls.append((done, total)))
    assert calls == [(2, 5), (4, 5), (5, 5)] and len(index) == 5
