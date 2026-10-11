#!/usr/bin/env python
"""Evaluation harness (CLAUDE.md §7).

    python scripts/eval.py --retrieval hybrid --model openai/gpt-oss-120b --provider openai_compat
    python scripts/eval.py --retrieval dense --limit 10
    python scripts/eval.py --dry-run --limit 5 --embedder hash           # mocked provider, no network

For every question: retrieve (hybrid or dense only) → answer with the configured provider through
``docs_chat.service.answer`` (same prompt, parsing and pricing as the app) → judge the answer against
the reference with an LLM (correct / partial / wrong + one-sentence reason) → record cost and latency.
Unanswerable questions are scored by abstention (the exact "I couldn't find this…" sentence).

Outputs: eval/results/<config>.jsonl (one row per question), .json (config + summary), .md (tables).
Rate limits: ``--delay`` seconds between API calls and retries with backoff on HTTP 429.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docs_chat import service  # noqa: E402
from docs_chat.chunking import chunk_pages  # noqa: E402
from docs_chat.config import settings  # noqa: E402
from docs_chat.costs import cost_usd  # noqa: E402
from docs_chat.embeddings import get_embedder  # noqa: E402
from docs_chat.evaluation import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM,
    MockProvider,
    abstained,
    call_with_retry,
    hit_at_k,
    is_quota_exhausted,
    judge_messages,
    markdown_report,
    parse_judge,
    summarize,
)
from docs_chat.index import DocIndex  # noqa: E402
from docs_chat.loaders import load_folder  # noqa: E402
from docs_chat.providers import ProviderError, get_provider  # noqa: E402
from docs_chat.retriever import RETRIEVAL_MODES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXIT_OK, EXIT_NOT_CONFIGURED = 0, 2


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-").lower() or "model"


def evaluate_question(q: dict, index: DocIndex, embedder, cfg, *, mode, model, provider, judge_model,
                      judge_provider, retries, sleep, pause) -> dict:
    """One row of the results: retrieval hits, answer, abstention, judge verdict, cost and latency."""
    row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"], "doc": q["doc"], "gold_page": q.get("page")}
    try:
        record = call_with_retry(
            lambda: service.answer(index, q["question"], [], model, provider, embedder=embedder, cfg=cfg, mode=mode),
            retries=retries, sleep=sleep,
        )
    except ProviderError as exc:
        row["error"] = str(exc)
        return row
    pause()
    passages = [(p.doc_name, p.page_no) for p in record.passages]
    row.update(
        retrieved=passages,
        hit3=hit_at_k(passages, q["doc"], q.get("page"), 3),
        hit6=hit_at_k(passages, q["doc"], q.get("page"), 6),
        best_dense_score=record.best_dense_score,
        best_bm25_score=record.best_bm25_score,
        weak=record.weak_retrieval,
        answer=record.text,
        citations=record.citations,
        invalid_citations=record.invalid_citations,
        abstained=abstained(record.text),
        cost_usd=record.cost_usd,
        judge_cost_usd=0.0,
        latency_s=record.latency_seconds,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        verdict=None,
        judge_reason=None,
    )
    if q["answerable"]:
        try:
            verdict = call_with_retry(
                lambda: judge_provider.complete(
                    judge_messages(q["question"], q["expected_answer"], record.text), JUDGE_SYSTEM, judge_model, JUDGE_MAX_TOKENS
                ),
                retries=retries, sleep=sleep,
            )
            pause()
            row["verdict"], row["judge_reason"] = parse_judge(verdict.text)
            row["judge_raw"] = verdict.text[:500]
            row["judge_cost_usd"] = cost_usd(
                judge_model, verdict.input_tokens, verdict.output_tokens, verdict.cache_read_tokens, provider=judge_provider.name
            )
        except ProviderError as exc:
            row["verdict"], row["judge_reason"] = "unknown", f"judge call failed: {exc}"
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval", choices=RETRIEVAL_MODES, default="hybrid")
    parser.add_argument("--provider", choices=["openai_compat", "anthropic"], default=settings.llm_provider)
    parser.add_argument("--model", default=settings.llm_model, help="answering model (default: LLM_MODEL)")
    parser.add_argument("--judge-model", default=None, help="judge model (default: the answering model)")
    parser.add_argument("--judge-provider", choices=["openai_compat", "anthropic"], default=None)
    parser.add_argument("--embedder", choices=["e5", "hash"], default=None, help="default: EMBEDDER (hash in --dry-run)")
    parser.add_argument("--index", type=Path, default=None, help="load this index instead of building one from --docs")
    parser.add_argument("--docs", type=Path, default=ROOT / "eval" / "sample_docs")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (quick runs)")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between API calls (free-tier rate limits)")
    parser.add_argument("--retries", type=int, default=4, help="retries with backoff on HTTP 429")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "results")
    parser.add_argument("--name", default=None, help="config name for the output files (default: <retrieval>-<model>)")
    parser.add_argument("--dry-run", action="store_true", help="mocked provider: answers from the eval set, no network")
    args = parser.parse_args(argv)

    embedder_kind = args.embedder or ("hash" if args.dry_run else settings.embedder)
    cfg = replace(settings, embedder=embedder_kind, top_k=args.top_k)
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    if args.dry_run:
        model = args.model or "mock-model"
        judge_model = args.judge_model or model
        provider = judge_provider = MockProvider(questions)
        provider_name = judge_provider_name = "mock"
        delay = 0.0
    else:
        model = args.model
        if not model:
            print("error: no model: pass --model or set LLM_MODEL in .env", file=sys.stderr)
            return EXIT_NOT_CONFIGURED
        judge_model = args.judge_model or model
        provider_name, judge_provider_name = args.provider, args.judge_provider or args.provider
        try:
            provider = get_provider(replace(settings, llm_provider=provider_name))
            judge_provider = provider if judge_provider_name == provider_name else get_provider(
                replace(settings, llm_provider=judge_provider_name)
            )
            cost_usd(model, 0, 0, provider=provider.name)  # a paid provider needs a price for the model
            cost_usd(judge_model, 0, 0, provider=judge_provider.name)
        except (ValueError, KeyError) as exc:
            print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
            return EXIT_NOT_CONFIGURED
        delay = args.delay

    t0 = time.perf_counter()
    embedder = get_embedder(cfg.embedder, cfg.e5_model_name)
    embedder.load()
    if args.index:
        index = DocIndex.load(args.index)
    else:
        pages, warnings = load_folder(args.docs)
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
        index = DocIndex.build(chunk_pages(pages, cfg.chunk_words, cfg.chunk_overlap_words), embedder)
    print(f"index: {len(index)} chunks, embedder {embedder.name}, ready in {time.perf_counter() - t0:.1f} s", flush=True)

    name = args.name or f"{args.retrieval}-{slug(model)}" + ("-dryrun" if args.dry_run else "")
    sleep = (lambda s: None) if args.dry_run else time.sleep
    pause = (lambda: sleep(delay)) if delay else (lambda: None)
    print(f"config {name}: {len(questions)} questions, retrieval={args.retrieval}, model={model} via {provider_name}, "
          f"judge={judge_model} via {judge_provider_name}, delay={delay}s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / f"{name}.jsonl"
    jsonl_path.write_text("", encoding="utf-8")  # rows are appended as they come, so an interrupted run keeps them
    rows: list[dict] = []
    stopped = None
    for i, q in enumerate(questions, start=1):
        row = evaluate_question(
            q, index, embedder, cfg, mode=args.retrieval, model=model, provider=provider, judge_model=judge_model,
            judge_provider=judge_provider, retries=args.retries, sleep=sleep, pause=pause,
        )
        rows.append(row)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row.get("error"):
            status = f"ERROR {row['error'][:70]}"
        elif q["answerable"]:
            status = f"hit@6={'✓' if row['hit6'] else '✗'} verdict={row['verdict']}"
        else:
            status = f"abstained={'✓' if row['abstained'] else '✗'}"
        print(f"[{i:2d}/{len(questions)}] {q['id']} {status}", flush=True)
        if row.get("error") and is_quota_exhausted(ProviderError(row["error"], 429)):
            stopped = f"stopped after {i} question(s): the provider's daily quota is exhausted ({row['error'][:120]})"
            print(stopped, flush=True)
            break

    summary = summarize(rows)
    config = {
        "name": name, "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "retrieval": args.retrieval, "model": model, "provider": provider_name, "judge_model": judge_model,
        "judge_provider": judge_provider_name, "embedder": embedder.name, "top_k": cfg.top_k,
        "chunk_words": cfg.chunk_words, "chunk_overlap_words": cfg.chunk_overlap_words,
        "weak_retrieval_threshold": cfg.weak_retrieval_threshold, "weak_bm25_threshold": cfg.weak_bm25_threshold,
        "questions_file": str(args.questions), "n_chunks": len(index), "dry_run": args.dry_run,
        "stopped_early": stopped,
    }
    (args.out_dir / f"{name}.json").write_text(json.dumps({"config": config, "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    report = markdown_report(name, config, summary, rows)
    (args.out_dir / f"{name}.md").write_text(report, encoding="utf-8")
    print()
    print(report.split("## Per question")[0].rstrip())
    print(f"\nwritten: {args.out_dir / name}.{{jsonl,json,md}}", flush=True)
    return EXIT_OK if stopped is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
