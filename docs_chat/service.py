"""The glue between the UI (or the CLI) and the pipeline — no Gradio in here.

* ``shared_embedder()`` loads the embedding model once per process and shares it; the app
  calls it at start-up (~8 s for e5-small on CPU) so no request pays for the load.
* ``build_index(session, file_paths, progress_cb)`` replaces the session's uploads with the
  given files and runs load → chunk → embed → save through the session, which enforces the
  limits. Returns a ``BuildReport`` (documents with page counts, loader warnings, timings).
* ``ask(session, question, history, model, provider)`` is a generator: it yields the answer
  text as it streams and *returns* an ``AnswerRecord`` (text, citations, sources, cost string,
  latency, weak-retrieval flag). It checks the configuration and the daily budget first,
  counts the question against the session limit, records the cost afterwards, and logs
  only counts, tokens and timings — never the question or the answer.

Errors callers should expect: ``ValueError`` (including ``LimitExceeded``) for configuration
and input problems, ``BudgetExceeded`` when the daily cap is reached, ``ProviderError`` when
the API call fails. ``scripts/ask.py`` and ``app.py`` both go through this module.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .config import settings as default_settings
from .costs import BudgetGuard, cost_usd, format_cost
from .embeddings import Embedder, get_embedder
from .llm import answer_stream, build_messages, parse_citations
from .providers import LLMProvider
from .retriever import retrieve
from .session import BuildReport, ProgressCallback, Session, report_progress

log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """The daily demo budget is spent; the message is safe to show to the user."""


# --------------------------------------------------------------------------- embedder
_shared_embedder: Embedder | None = None
_shared_lock = threading.Lock()


def shared_embedder(cfg: Settings = default_settings) -> Embedder:
    """The process-wide embedder, created and loaded on the first call (the first call's settings win)."""
    global _shared_embedder
    with _shared_lock:
        if _shared_embedder is None:
            t0 = time.perf_counter()
            embedder = get_embedder(cfg.embedder, cfg.e5_model_name)
            embedder.load()
            _shared_embedder = embedder
            log.info("embedder %s loaded in %.1f s", embedder.name, time.perf_counter() - t0)
        return _shared_embedder


def reset_shared_embedder() -> None:
    """Forget the shared embedder (tests)."""
    global _shared_embedder
    with _shared_lock:
        _shared_embedder = None


# ----------------------------------------------------------------------------- index
def build_index(
    session: Session,
    file_paths: Iterable[str | Path],
    progress_cb: ProgressCallback | None = None,
    *,
    embedder: Embedder | None = None,
) -> BuildReport:
    """Replace the session's uploads with ``file_paths`` and build its index (limits enforced by the session)."""
    paths = [Path(p) for p in file_paths]
    if not paths:
        raise ValueError("Upload at least one document first.")
    report_progress(progress_cb, 0.0, f"Copying {len(paths)} file(s)…")
    session.clear_uploads()
    session.add_uploads(paths)
    return session.build_index(embedder or shared_embedder(session.settings), progress_cb=progress_cb)


# ---------------------------------------------------------------------------- answers
@dataclass(frozen=True)
class Source:
    """A cited passage, for the UI's "Sources" list."""

    n: int  # the [n] in the answer
    doc_name: str
    page_no: int
    is_pseudo_page: bool
    text: str

    @property
    def where(self) -> str:
        return f"section {self.page_no}" if self.is_pseudo_page else f"p.{self.page_no}"


@dataclass(frozen=True)
class AnswerRecord:
    text: str  # normalised answer (ASCII [n] citations, plain spaces)
    citations: list[int]  # passage numbers cited that exist
    invalid_citations: int  # citations to non-existent passages, dropped
    sources: list[Source]
    cost: str  # "free tier", "$0.0004", "$0.0004 (est.)"
    cost_usd: float
    latency_seconds: float  # retrieval + answer, model already loaded
    weak_retrieval: bool
    best_dense_score: float | None
    n_passages: int
    model: str
    truncated: bool  # the answer hit the max_tokens limit


def budget_guard(cfg: Settings = default_settings) -> BudgetGuard:
    return BudgetGuard(cfg.budget_file, cfg.max_daily_usd)


def preflight(model: str, provider: LLMProvider, guard: BudgetGuard) -> None:
    """Raise before any API call if the model is not configured or priced, or the daily budget is spent."""
    if not model:
        raise ValueError("No LLM model configured: set LLM_MODEL in .env (see .env.example)")
    try:
        cost_usd(model, 0, 0, provider=provider.name)  # a paid provider needs a price for the model
    except KeyError as exc:
        raise ValueError(str(exc.args[0])) from None
    if not guard.can_spend():
        spend = guard.read()
        raise BudgetExceeded(
            f"Demo budget reached for today: ${spend.usd:.4f} spent in {spend.calls} call(s), "
            f"cap ${guard.max_daily_usd:.2f}. Not calling the API."
        )


def ask(
    session: Session,
    question: str,
    history: Sequence[tuple[str, str]],
    model: str,
    provider: LLMProvider,
    *,
    guard: BudgetGuard | None = None,
    embedder: Embedder | None = None,
    cfg: Settings | None = None,
) -> Generator[str, None, AnswerRecord]:
    """Yield the answer as it streams; the generator's return value is the AnswerRecord.

    Raises (on the first ``next()``): ValueError when there is no index or question, or the
    model is not configured; BudgetExceeded; LimitExceeded once the question limit is hit;
    ProviderError if the API call fails (nothing is charged to the budget then).
    """
    cfg = cfg or session.settings
    guard = guard or budget_guard(cfg)
    if session.index is None:
        raise ValueError("No index yet: upload documents and build the index first.")
    question = question.strip()
    if not question:
        raise ValueError("Please type a question.")
    preflight(model, provider, guard)
    embedder = embedder or shared_embedder(cfg)

    t0 = time.perf_counter()
    retrieval = retrieve(session.index, question, embedder, cfg)
    hits = retrieval.hits
    messages = build_messages(history, hits, question, weak_retrieval=retrieval.weak)
    session.next_question()  # LimitExceeded once max_questions_per_session is reached
    result = yield from answer_stream(provider, messages, model)
    latency = time.perf_counter() - t0

    usd = cost_usd(model, result.input_tokens, result.output_tokens, result.cache_read_tokens, provider=provider.name)
    guard.record(usd)  # free-tier calls record $0 but still count
    text, cited, invalid = parse_citations(result.text, len(hits))
    sources = [
        Source(n, hit.chunk.doc_name, hit.chunk.page_no, hit.chunk.is_pseudo_page, hit.chunk.text)
        for n in cited
        for hit in [hits[n - 1]]
    ]
    cost = format_cost(usd, estimated=result.usage_estimated)
    log.info(
        "answer: %d passages, %d in / %d out tokens%s, %.1f s, cost %s, %d source(s), %d invalid citation(s)%s",
        len(hits), result.input_tokens, result.output_tokens, " (estimated)" if result.usage_estimated else "",
        latency, cost, len(sources), invalid, ", weak retrieval" if retrieval.weak else "",
    )
    return AnswerRecord(
        text=text,
        citations=cited,
        invalid_citations=invalid,
        sources=sources,
        cost=cost,
        cost_usd=usd,
        latency_seconds=latency,
        weak_retrieval=retrieval.weak,
        best_dense_score=retrieval.best_dense_score,
        n_passages=len(hits),
        model=result.model,
        truncated=result.stop_reason in ("length", "max_tokens"),
    )
