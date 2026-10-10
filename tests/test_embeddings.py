from __future__ import annotations

from types import SimpleNamespace as NS

from docs_chat.embeddings import _embedding_dim, get_embedder


def test_embedding_dim_prefers_the_new_sentence_transformers_method():
    new_api = NS(get_embedding_dimension=lambda: 384, get_sentence_embedding_dimension=lambda: 0)
    old_api = NS(get_sentence_embedding_dimension=lambda: 384)  # sentence-transformers < 6
    assert _embedding_dim(new_api) == 384
    assert _embedding_dim(old_api) == 384


def test_hash_embedder_load_is_a_no_op():
    embedder = get_embedder("hash")
    embedder.load()
    assert (embedder.name, embedder.dim) == ("hash-256", 256)
    assert embedder.encode_queries(["x"]).shape == (1, 256)
