#!/usr/bin/env python
"""Milestone 4 — evaluation harness (CLAUDE.md §7).

Target usage:
    python scripts/eval.py --config hybrid --model claude-haiku-5-5
    python scripts/eval.py --all            # dense / hybrid / hybrid+rerank × Haiku / Sonnet

Metrics:
  1. retrieval hit rate@3 and @6 (gold page among retrieved chunks) — implemented below,
     works without an API key, so retrieval can be tuned first;
  2. answer correctness by an LLM judge (claude-sonnet-5-5, 3-level rubric) — TODO M4;
  3. abstention accuracy on unanswerable questions — TODO M4;
  4. mean / p50 cost per answer and p50 latency — TODO M4.
Writes a markdown table + JSON to eval/results/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docs_chat.chunking import chunk_pages  # noqa: E402
from docs_chat.config import settings  # noqa: E402
from docs_chat.embeddings import get_embedder  # noqa: E402
from docs_chat.index import DocIndex  # noqa: E402
from docs_chat.loaders import load_folder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieval_hit_rate(index: DocIndex, embedder, questions: list[dict], ks=(3, 6)) -> dict[str, float]:
    answerable = [q for q in questions if q["answerable"]]
    hits = {k: 0 for k in ks}
    for q in answerable:
        found = index.hybrid_search(q["question"], embedder, k=max(ks), candidates=settings.candidates, rrf_k=settings.rrf_k)
        for k in ks:
            if any(h.chunk.doc_name == q["doc"] and h.chunk.page_no == q["page"] for h in found[:k]):
                hits[k] += 1
    return {f"hit@{k}": hits[k] / max(len(answerable), 1) for k in ks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, default=ROOT / "eval" / "sample_docs")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    parser.add_argument("--embedder", default=settings.embedder, choices=["e5", "hash"])
    parser.add_argument("--retrieval-only", action="store_true", default=True, help="(default until M4)")
    args = parser.parse_args(argv)

    pages, warnings = load_folder(args.docs)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    embedder = get_embedder(args.embedder, settings.e5_model_name)
    index = DocIndex.build(chunk_pages(pages, settings.chunk_words, settings.chunk_overlap_words), embedder)
    questions = load_questions(args.questions)
    metrics = retrieval_hit_rate(index, embedder, questions)
    print(f"embedder={embedder.name} chunks={len(index)} questions={len(questions)}")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"retrieval_{embedder.name.replace('/', '_')}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # TODO (M4): answer correctness (LLM judge), abstention accuracy, cost, latency, markdown table.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
