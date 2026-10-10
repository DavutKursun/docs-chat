#!/usr/bin/env python
"""Ask a question against an index from the command line.

    python scripts/ask.py --index index/ "How many vacation days do new employees get?"

Flow: load index + embedder (reported on its own line, not part of the answer latency) →
hybrid retrieval → prompt → answer streamed to the terminal → the cited passages (doc, page,
first 200 characters) → "cost $0.0004 · 2.1 s · 6 passages".

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docs_chat.config import settings  # noqa: E402
from docs_chat.costs import BudgetGuard, cost_usd, format_cost  # noqa: E402
from docs_chat.embeddings import get_embedder  # noqa: E402
from docs_chat.index import DocIndex, Hit  # noqa: E402
from docs_chat.llm import answer_stream, build_messages, parse_citations  # noqa: E402
from docs_chat.providers import LLMProvider, LLMResult, Message, ProviderError, get_provider  # noqa: E402
from docs_chat.retriever import retrieve  # noqa: E402

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


def stream_answer(provider: LLMProvider, messages: list[Message], model: str) -> LLMResult:
    """Print the answer as it streams; return the provider's result (text, usage, timing)."""
    gen = answer_stream(provider, messages, model)
    printed = False
    try:
        while True:
            try:
                delta = next(gen)
            except StopIteration as stop:
                return stop.value
            print(delta, end="", flush=True)
            printed = True
    finally:
        if printed:  # end the answer line, also when the stream fails half-way
            print(flush=True)


def print_sources(hits: list[Hit], cited: list[int], invalid: int) -> None:
    print()
    print("Sources:" if cited else "Sources: none cited")
    for n in cited:
        c = hits[n - 1].chunk
        where = f"section {c.page_no}" if c.is_pseudo_page else f"p.{c.page_no}"
        print(f"  [{n}] {c.doc_name} {where}: {preview(c.text)}")
    print(f"invalid citations dropped: {invalid}")


def main(argv: list[str] | None = None, *, provider: LLMProvider | None = None, guard: BudgetGuard | None = None) -> int:
    """``provider`` and ``guard`` are injectable for tests; by default both come from the settings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--embedder", default=settings.embedder, choices=["e5", "hash"])
    parser.add_argument("--model", default=settings.llm_model, help="LLM model id (default: LLM_MODEL)")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM; print the retrieved passages")
    args = parser.parse_args(argv)

    if not args.retrieval_only:
        # fail fast on configuration and budget, before loading the index and the embedding model
        if provider is None:
            try:
                provider = get_provider(settings)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_NOT_CONFIGURED
        if not args.model:
            print("error: No LLM model configured: set LLM_MODEL in .env (see .env.example)", file=sys.stderr)
            return EXIT_NOT_CONFIGURED
        try:
            cost_usd(args.model, 0, 0, provider=provider.name)  # a paid provider needs a price for the model
        except KeyError as exc:
            print(f"error: {exc.args[0]}", file=sys.stderr)
            return EXIT_NOT_CONFIGURED
        guard = guard or BudgetGuard(settings.budget_file, settings.max_daily_usd)
        if not guard.can_spend():
            spend = guard.read()
            print(
                f"Demo budget reached for today: ${spend.usd:.4f} spent in {spend.calls} call(s), "
                f"cap ${guard.max_daily_usd:.2f}. Not calling the API.",
                file=sys.stderr,
            )
            return EXIT_BUDGET

    t_load = time.perf_counter()
    index = DocIndex.load(args.index)
    embedder = get_embedder(args.embedder, settings.e5_model_name)
    if embedder.name != index.embedder_name:
        print(f"warning: index was built with {index.embedder_name}, querying with {embedder.name}", file=sys.stderr)
    embedder.load()  # load the model now, so the seconds below are answer latency only
    print(f"loaded index ({len(index)} chunks) + embedder {embedder.name} in {time.perf_counter() - t_load:.1f} s")

    t0 = time.perf_counter()
    result = retrieve(index, args.question, embedder, settings)
    hits = result.hits
    if args.retrieval_only:
        print_retrieval(hits, result.weak, result.best_dense_score, time.perf_counter() - t0)
        return EXIT_OK

    if result.weak:
        best = None if result.best_dense_score is None else round(result.best_dense_score, 3)
        print(f"note: retrieval was weak (best dense score {best}); the answer may not be in the documents.")
    messages = build_messages([], hits, args.question, weak_retrieval=result.weak)
    try:
        llm = stream_answer(provider, messages, args.model)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_API_ERROR
    seconds = time.perf_counter() - t0

    usd = cost_usd(args.model, llm.input_tokens, llm.output_tokens, llm.cache_read_tokens, provider=provider.name)
    guard.record(usd)  # free-tier calls record $0 but still count as calls

    _, cited, invalid = parse_citations(llm.text, len(hits))  # llm.text is already normalised by answer_stream
    print_sources(hits, cited, invalid)
    if llm.stop_reason in ("length", "max_tokens"):
        print("note: the answer was cut off by the max_tokens limit", file=sys.stderr)
    print(f"cost {format_cost(usd, estimated=llm.usage_estimated)} · {seconds:.1f} s · {len(hits)} passages")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
