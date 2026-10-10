"""Evaluation metrics and helpers behind ``scripts/eval.py`` (CLAUDE.md §7) — pure functions, testable offline.

* ``hit_at_k`` — is the gold page among the top-k retrieved passages of the gold document?
* ``abstained`` — does the answer contain the exact abstention sentence?
* the LLM judge: ``JUDGE_SYSTEM``, ``judge_messages``, ``parse_judge`` (correct / partial / wrong + one-sentence reason)
* ``summarize`` — hit@3/6, correctness, abstention accuracy, invalid-citation rate, cost and latency percentiles
* ``markdown_report`` — the summary as a markdown table plus a per-question table
* ``MockProvider`` — a deterministic provider for ``--dry-run`` and tests (answers from the eval set itself)
* ``call_with_retry`` / ``is_rate_limited`` — retry with backoff on HTTP 429 (Groq free tier)
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from .costs import format_cost
from .llm import NOT_FOUND_SENTENCE, normalize_answer
from .providers import LLMResult, Message, ProviderError

VERDICTS = ("correct", "partial", "wrong")
JUDGE_MAX_TOKENS = 400  # the judge's JSON line is short, but reasoning models spend tokens before it

JUDGE_SYSTEM = """You grade answers given by a document question-answering system.
You receive the question, a reference answer written by a human, and the system's answer.
Decide:
- "correct": the system's answer states the same facts as the reference; wording and language may differ and extra correct detail is fine.
- "partial": it states part of the reference facts, or is right but materially incomplete or hedged.
- "wrong": it contradicts the reference, answers a different question, invents facts, or says the information is not available although the reference gives it.
Judge facts only, not style or citation markers like [1].
Reply with exactly one line of JSON: {"verdict": "correct" | "partial" | "wrong", "reason": "<one sentence>"}"""


# ------------------------------------------------------------------------- metrics
def hit_at_k(passages: Sequence[tuple[str, int]], gold_doc: str, gold_page: int | None, k: int) -> bool:
    """True if the gold (doc, page) appears among the first k retrieved passages."""
    if gold_page is None:
        return False
    return any(doc == gold_doc and page == gold_page for doc, page in passages[:k])


def abstained(answer: str) -> bool:
    """True if the exact abstention sentence is in the answer (apostrophe variants tolerated)."""
    text = normalize_answer(answer).replace("’", "'")
    return NOT_FOUND_SENTENCE in text


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile, p in 0..100; None when there are no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rate(flags: Sequence[bool]) -> float | None:
    return sum(bool(f) for f in flags) / len(flags) if flags else None


# --------------------------------------------------------------------------- judge
def judge_messages(question: str, reference: str, answer: str) -> list[Message]:
    return [{"role": "user", "content": f"Question: {question}\n\nReference answer: {reference}\n\nSystem answer: {answer}"}]


_VERDICT = re.compile(r'"verdict"\s*:\s*"(correct|partial|wrong)"', re.IGNORECASE)
_REASON = re.compile(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', re.IGNORECASE | re.DOTALL)
_WORDS = {"correct": re.compile(r"\bcorrect\b"), "partial": re.compile(r"\bpartial"), "wrong": re.compile(r"\bwrong\b")}


def parse_judge(text: str) -> tuple[str, str]:
    """(verdict, reason) from the judge's reply; verdict is "unknown" when it cannot be read."""
    match = _VERDICT.search(text)
    if match:
        reason = _REASON.search(text)
        return match.group(1).lower(), (reason.group(1).replace('\\"', '"').strip() if reason else "")
    lowered = text.lower()
    found = [(m.start(), verdict) for verdict, rx in _WORDS.items() if (m := rx.search(lowered))]
    if found:  # fall back to the first rubric word that appears
        return min(found)[1], text.strip()[:200]
    return "unknown", text.strip()[:200]


# ------------------------------------------------------------------------- summary
def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question rows (see scripts/eval.py) into the metrics of CLAUDE.md §7."""
    ok = [r for r in rows if not r.get("error")]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    judged = [r for r in answerable if r.get("verdict") in VERDICTS]
    costs = [r["cost_usd"] for r in ok]
    latencies = [r["latency_s"] for r in ok]
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "n_errors": len(rows) - len(ok),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "hit@3": rate([r["hit3"] for r in answerable]),
        "hit@6": rate([r["hit6"] for r in answerable]),
        "correct": sum(r["verdict"] == "correct" for r in judged),
        "partial": sum(r["verdict"] == "partial" for r in judged),
        "wrong": sum(r["verdict"] == "wrong" for r in judged),
        "unknown_verdicts": len(answerable) - len(judged),
        "correct_rate": rate([r["verdict"] == "correct" for r in judged]),
        "correct_or_partial_rate": rate([r["verdict"] in ("correct", "partial") for r in judged]),
        "abstention_accuracy": rate([r["abstained"] for r in unanswerable]),
        "false_abstention_rate": rate([r["abstained"] for r in answerable]),
        "invalid_citation_rate": rate([r["invalid_citations"] > 0 for r in ok]),
        "invalid_citations_total": sum(r["invalid_citations"] for r in ok),
        "weak_retrieval_rate": rate([r["weak"] for r in ok]),
        "cost_mean_usd": mean(costs),
        "cost_p50_usd": percentile(costs, 50),
        "cost_total_usd": sum(costs),
        "judge_cost_total_usd": sum(r.get("judge_cost_usd", 0.0) for r in ok),
        "latency_mean_s": mean(latencies),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def _cost(value: float | None) -> str:
    return "n/a" if value is None else format_cost(value)


def markdown_report(name: str, config: dict[str, Any], summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    s = summary
    lines = [
        f"# Evaluation: {name}",
        "",
        f"Generated {config.get('generated_at', '')} · retrieval **{config.get('retrieval')}** · "
        f"model `{config.get('model')}` via {config.get('provider')} · judge `{config.get('judge_model')}` via "
        f"{config.get('judge_provider')} · embedder `{config.get('embedder')}` · "
        f"{s['n']} questions ({s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable"
        + (f", {s['n_errors']} failed" if s["n_errors"] else "")
        + ")",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Retrieval hit@3 | {_pct(s['hit@3'])} |",
        f"| Retrieval hit@6 | {_pct(s['hit@6'])} |",
        f"| Judge: correct / partial / wrong | {s['correct']} / {s['partial']} / {s['wrong']}"
        + (f" ({s['unknown_verdicts']} unreadable)" if s["unknown_verdicts"] else "")
        + " |",
        f"| Correct + partial | {_pct(s['correct_or_partial_rate'])} |",
        f"| Abstention accuracy ({s['n_unanswerable']} unanswerable) | {_pct(s['abstention_accuracy'])} |",
        f"| False abstentions (answerable) | {_pct(s['false_abstention_rate'])} |",
        f"| Answers with invalid citations | {_pct(s['invalid_citation_rate'])} ({s['invalid_citations_total']} dropped) |",
        f"| Weak-retrieval flag rate | {_pct(s['weak_retrieval_rate'])} |",
        f"| Cost per answer, mean / p50 | {_cost(s['cost_mean_usd'])} / {_cost(s['cost_p50_usd'])} |",
        f"| Cost total (answers + judge) | {_cost(s['cost_total_usd'])} + {_cost(s['judge_cost_total_usd'])} |",
        f"| Latency p50 / p95 | {_secs(s['latency_p50_s'])} / {_secs(s['latency_p95_s'])} |",
        "",
        "## Per question",
        "",
        "| id | type | hit@3 | hit@6 | verdict | abstained | invalid cit. | cost | latency | reason |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['id']} | {'A' if r['answerable'] else 'U'} | – | – | error | – | – | – | – | {r['error'][:80]} |")
            continue
        verdict = r.get("verdict") or ("abstained" if r["abstained"] else "answered")
        lines.append(
            f"| {r['id']} | {'A' if r['answerable'] else 'U'} | {'✓' if r['hit3'] else '✗'} | {'✓' if r['hit6'] else '✗'} | "
            f"{verdict} | {'yes' if r['abstained'] else 'no'} | {r['invalid_citations']} | {_cost(r['cost_usd'])} | "
            f"{_secs(r['latency_s'])} | {(r.get('judge_reason') or '').replace('|', '/')[:90]} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- mock + retry
class MockProvider:
    """Deterministic provider for ``--dry-run`` and tests: answers from the eval set, judges everything correct."""

    name = "openai_compat"  # unpriced model → free tier

    def __init__(self, questions: Sequence[dict[str, Any]]) -> None:
        self._by_question = {q["question"]: q for q in questions}
        self.calls = 0

    def _reply(self, messages: Sequence[Message], system: str) -> str:
        if system == JUDGE_SYSTEM:
            return '{"verdict": "correct", "reason": "dry run: not judged"}'
        match = re.search(r"^Question: (.+)$", messages[-1]["content"], re.MULTILINE)
        q = self._by_question.get(match.group(1).strip()) if match else None
        if q is None or not q["answerable"]:
            return f"{NOT_FOUND_SENTENCE} Information about this topic would be needed."
        return f"{q['expected_answer']} [1]"

    def complete(self, messages: Sequence[Message], system: str, model: str, max_tokens: int) -> LLMResult:
        self.calls += 1
        return LLMResult(
            text=self._reply(messages, system), input_tokens=1500, output_tokens=60, cache_read_tokens=0,
            model=model, elapsed_seconds=0.0, usage_estimated=False, stop_reason="stop",
        )

    def stream(self, messages, system, model, max_tokens):
        result = self.complete(messages, system, model, max_tokens)
        yield result.text
        return result


def is_rate_limited(exc: BaseException) -> bool:
    text = str(exc).lower()
    return getattr(exc, "status_code", None) == 429 or "429" in text or "rate limit" in text or "rate_limit" in text


T = TypeVar("T")
BACKOFF_S = (15, 30, 60, 60)


def call_with_retry(fn: Callable[[], T], retries: int = 4, sleep: Callable[[float], None] = time.sleep) -> T:
    """Call ``fn``; on a rate-limit ProviderError wait (15, 30, 60, 60 s) and retry; other errors propagate."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except ProviderError as exc:
            if attempt >= retries or not is_rate_limited(exc):
                raise
            sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
    raise AssertionError("unreachable")
