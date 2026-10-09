#!/usr/bin/env python
"""Build a search index from a folder of documents.

Usage:
    python scripts/ingest.py eval/sample_docs --out index/
    EMBEDDER=hash python scripts/ingest.py eval/sample_docs --out index/   # offline smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docs_chat.chunking import chunk_pages  # noqa: E402
from docs_chat.config import settings  # noqa: E402
from docs_chat.embeddings import get_embedder  # noqa: E402
from docs_chat.index import DocIndex  # noqa: E402
from docs_chat.loaders import load_folder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with .pdf/.docx/.txt/.md files")
    parser.add_argument("--out", type=Path, default=Path("index"), help="output directory")
    parser.add_argument("--embedder", default=settings.embedder, choices=["e5", "hash"])
    parser.add_argument("--chunk-words", type=int, default=settings.chunk_words)
    parser.add_argument("--overlap-words", type=int, default=settings.chunk_overlap_words)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    pages, warnings = load_folder(args.folder)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if not pages:
        print("Nothing to index.", file=sys.stderr)
        return 1
    chunks = chunk_pages(pages, chunk_words=args.chunk_words, overlap_words=args.overlap_words)
    t1 = time.perf_counter()

    embedder = get_embedder(args.embedder, settings.e5_model_name)
    index = DocIndex.build(chunks, embedder)
    index.save(args.out)
    t2 = time.perf_counter()

    docs = index.documents
    print(f"Indexed {len(docs)} document(s), {len(pages)} page(s), {len(chunks)} chunk(s) -> {args.out}/")
    for name, n_pages in docs.items():
        print(f"  - {name}: {n_pages} page(s) with text")
    print(f"load+chunk {t1 - t0:.1f}s, embed+index {t2 - t1:.1f}s (embedder: {embedder.name}, dim {index.dim})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
