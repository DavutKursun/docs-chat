#!/usr/bin/env python
"""Milestone 2 — ask a question against an index from the command line.

Target usage (implement in M2):
    python scripts/ask.py --index index/ "How many vacation days do new employees get?"

Prints: the streamed answer with [n] citations, then the cited passages
(doc, page, first 200 chars), then "cost $0.0004 · 2.1 s · 6 passages".

Until M2 is done, this script only runs retrieval and prints the top passages, which
is useful to eyeball retrieval quality without an API key.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docs_chat.config import settings  # noqa: E402
from docs_chat.embeddings import get_embedder  # noqa: E402
from docs_chat.index import DocIndex  # noqa: E402
from docs_chat.retriever import retrieve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--embedder", default=settings.embedder, choices=["e5", "hash"])
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM (default until M2)")
    args = parser.parse_args(argv)

    index = DocIndex.load(args.index)
    embedder = get_embedder(args.embedder, settings.e5_model_name)
    if embedder.name != index.embedder_name:
        print(f"warning: index was built with {index.embedder_name}, querying with {embedder.name}", file=sys.stderr)

    t0 = time.perf_counter()
    result = retrieve(index, args.question, embedder, settings)
    dt = time.perf_counter() - t0
    print(f"retrieval: {len(result.hits)} passages in {dt:.2f}s, best dense score "
          f"{result.best_dense_score if result.best_dense_score is None else round(result.best_dense_score, 3)}"
          f"{' (WEAK)' if result.weak else ''}\n")
    for i, hit in enumerate(result.hits, start=1):
        c = hit.chunk
        print(f"[{i}] {c.doc_name} p.{c.page_no}  rrf={hit.score:.4f} dense={hit.dense_score} bm25={hit.bm25_score}")
        print("    " + c.text[:200].replace("\n", " ") + ("…" if len(c.text) > 200 else ""))
    # TODO (M2): build messages, stream the answer, print citations + cost.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
