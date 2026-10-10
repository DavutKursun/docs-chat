#!/usr/bin/env python
"""Ask a question against an index from the command line.

    python scripts/ask.py --index index/ "How many vacation days do new employees get?"

Loads the index and the embedder (reported on its own line, not part of the answer latency),
then goes through ``docs_chat.service`` — the same code path as the web app: hybrid
retrieval → prompt → answer streamed to the terminal → the cited passages (doc, page, first
200 characters) → "cost $0.0004 · 2.1 s · 6 passages".

The daily budget guard (MAX_DAILY_USD, BUDGET_FILE) is checked before the call and every
call is recorded afterwards (free-tier calls record $0 but still count as calls).
``--retrieval-only`` stops after retrieval and prints the top passages with their scores;
it needs no API key and is useful to eyeball retrieval quality.

Exit codes: 0 ok · 2 provider or model not configured (or unpriced on a paid provider)
· 3 daily budget reached · 4 the API call failed (auth, unknown model, rate limit, connection).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docs_chat import service  # noqa: E402
from docs_chat.config import settings  # noqa: E402
from docs_chat.costs import BudgetGuard  # noqa: E402
from docs_chat.index import DocIndex, Hit  # noqa: E402
from docs_chat.providers import LLMProvider, ProviderError, get_provider  # noqa: E402
from docs_chat.retriever import retrieve  # noqa: E402
from docs_chat.session import Session  # noqa: E402

EXIT_OK, EXIT_NOT_CONFIGURED, EXIT_BUDGET, EXIT_API_ERROR = 0, 2, 3, 4
PREVIEW_CHARS = 200


def preview(text: str, n: int = PREVIEW_CHARS) -> str:
    text = text.replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


def print_retrieval(hits: list[Hit], weak: bool, best: float | None, seconds: float) -> None:
    """--retrieval-only output: every retrieved passage with its fusion, dense and BM25 scores."""
    best_text = best if best is None else round(best, 3)
    print(f"retrieval: {len(hits)} passages in {seconds:.2f}s, best dense score {best_text}{' (WEAK)' if weak else ''}\n")
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        print(f"[{i}] {c.doc_name} p.{c.page_no}  rrf={hit.score:.4f} dense={hit.dense_score} bm25={hit.bm25_score}")
        print("    " + preview(c.text))


def stream_answer(answer: Generator[str, None, service.AnswerRecord]) -> service.AnswerRecord:
    """Print the answer as it streams; return the record (text, sources, cost, latency)."""
    printed = False
    try:
        while True:
            try:
                delta = next(answer)
            except StopIteration as stop:
                return stop.value
            print(delta, end="", flush=True)
            printed = True
    finally:
        if printed:  # end the answer line, also when the stream fails half-way
            print(flush=True)


def print_sources(record: service.AnswerRecord) -> None:
    print()
    print("Sources:" if record.sources else "Sources: none cited")
    for source in record.sources:
        print(f"  [{source.n}] {source.doc_name} {source.where}: {preview(source.text)}")
    print(f"invalid citations dropped: {record.invalid_citations}")


def main(argv: list[str] | None = None, *, provider: LLMProvider | None = None, guard: BudgetGuard | None = None) -> int:
    """``provider`` and ``guard`` are injectable for tests; by default both come from the settings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--embedder", default=settings.embedder, choices=["e5", "hash"])
    parser.add_argument("--model", default=settings.llm_model, help="LLM model id (default: LLM_MODEL)")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM; print the retrieved passages")
    args = parser.parse_args(argv)
    cfg = replace(settings, embedder=args.embedder)

    if not args.retrieval_only:
        # fail fast on configuration and budget, before loading the index and the embedding model
        if provider is None:
            try:
                provider = get_provider(settings)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_NOT_CONFIGURED
        guard = guard or service.budget_guard(settings)
        try:
            service.preflight(args.model, provider, guard)
        except service.BudgetExceeded as exc:
            print(exc, file=sys.stderr)
            return EXIT_BUDGET
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_NOT_CONFIGURED

    t_load = time.perf_counter()
    index = DocIndex.load(args.index)
    embedder = service.shared_embedder(cfg)
    if embedder.name != index.embedder_name:
        print(f"warning: index was built with {index.embedder_name}, querying with {embedder.name}", file=sys.stderr)
    print(f"loaded index ({len(index)} chunks) + embedder {embedder.name} in {time.perf_counter() - t_load:.1f} s")

    if args.retrieval_only:
        t0 = time.perf_counter()
        result = retrieve(index, args.question, embedder, cfg)
        print_retrieval(result.hits, result.weak, result.best_dense_score, time.perf_counter() - t0)
        return EXIT_OK

    session = Session(id="cli", path=args.index, created_at=time.time(), settings=cfg, index=index)
    try:
        record = stream_answer(
            service.ask(session, args.question, [], args.model, provider, guard=guard, embedder=embedder)
        )
    except service.BudgetExceeded as exc:
        print(exc, file=sys.stderr)
        return EXIT_BUDGET
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_API_ERROR
    except ValueError as exc:  # not configured, or a session limit
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_NOT_CONFIGURED

    if record.weak_retrieval:
        best = None if record.best_dense_score is None else round(record.best_dense_score, 3)
        print(f"note: retrieval was weak (best dense score {best}); the answer may not be in the documents.")
    print_sources(record)
    if record.truncated:
        print("note: the answer was cut off by the max_tokens limit", file=sys.stderr)
    print(f"cost {record.cost} · {record.latency_seconds:.1f} s · {record.n_passages} passages")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
