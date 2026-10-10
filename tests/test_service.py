"""Service layer: index building with progress, answering through a mocked provider, guards and limits."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from docs_chat import service
from docs_chat.config import Settings
from docs_chat.costs import BudgetGuard
from docs_chat.embeddings import HashEmbedder
from docs_chat.llm import SYSTEM_PROMPT, WEAK_RETRIEVAL_NOTE
from docs_chat.providers import LLMResult, ProviderError
from docs_chat.session import LimitExceeded, SessionStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = sorted((ROOT / "eval" / "sample_docs").glob("*.pdf"))
QUESTION = "What does error code E3 mean?"
ANSWER = "Error code E3 means the device has overheated【1】. See also [9]."


class FakeProvider:
    """Streams a fixed answer word by word; optionally fails after ``after`` pieces."""

    name = "openai_compat"

    def __init__(self, text: str = ANSWER, error: Exception | None = None, after: int = 0):
        self.text, self.error, self.after, self.calls = text, error, after, []

    def stream(self, messages, system, model, max_tokens):
        self.calls.append({"messages": list(messages), "system": system, "model": model, "max_tokens": max_tokens})
        pieces = re.split(r"(?<= )", self.text)
        if self.error is not None:
            yield from pieces[: self.after]
            raise self.error
        yield from pieces
        return LLMResult(
            text=self.text, input_tokens=1500, output_tokens=40, cache_read_tokens=0,
            model=model, elapsed_seconds=0.02, usage_estimated=False, stop_reason="stop",
        )

    def complete(self, messages, system, model, max_tokens):
        raise NotImplementedError


def drain(gen):
    deltas = []
    try:
        while True:
            deltas.append(next(gen))
    except StopIteration as stop:
        return deltas, stop.value


def make_settings(**overrides) -> Settings:
    return Settings(**{"embedder": "hash", **overrides})


@pytest.fixture
def store(tmp_path):
    return SessionStore(tmp_path / "sessions", settings=make_settings())


@pytest.fixture
def guard(tmp_path):
    return BudgetGuard(tmp_path / "budget.json", max_daily_usd=2.0)


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def indexed_session(store, embedder):
    session = store.create()
    service.build_index(session, SAMPLE_DOCS, embedder=embedder)
    return session


# --- build_index -----------------------------------------------------------------------------


def test_build_index_reports_progress_documents_and_warnings(store, embedder):
    session = store.create()
    calls: list[tuple[float, str]] = []
    report = service.build_index(session, SAMPLE_DOCS, lambda f, m: calls.append((f, m)), embedder=embedder)
    assert report.documents == {"aria_x200_manual.pdf": 2, "lumen_employee_handbook.pdf": 3} == session.documents
    assert report.n_pages == 5 and report.n_chunks == 9 and report.warnings == [] and report.seconds >= 0
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert any("Embedding" in m for _, m in calls) and "Indexed 2 document(s)" in calls[-1][1]
    assert session.index is not None and (session.index_dir / "faiss.index").exists()


def test_build_index_replaces_earlier_uploads(store, embedder):
    session = store.create()
    service.build_index(session, SAMPLE_DOCS, embedder=embedder)
    service.build_index(session, SAMPLE_DOCS[:1], embedder=embedder)
    assert [p.name for p in session.uploaded_files()] == [SAMPLE_DOCS[0].name]
    assert list(session.documents) == [SAMPLE_DOCS[0].name]


def test_build_index_rejects_no_files_and_propagates_limits(store, embedder, tmp_path):
    session = store.create()
    with pytest.raises(ValueError, match="Upload at least one document"):
        service.build_index(session, [], embedder=embedder)
    tight = SessionStore(tmp_path / "tight", settings=make_settings(max_files=1)).create()
    with pytest.raises(LimitExceeded, match="Too many files"):
        service.build_index(tight, SAMPLE_DOCS, embedder=embedder)
    assert tight.index is None


# --- ask --------------------------------------------------------------------------------------


def test_ask_streams_deltas_and_returns_a_record(indexed_session, guard, embedder, caplog):
    provider = FakeProvider()
    with caplog.at_level(logging.INFO, logger="docs_chat.service"):
        deltas, record = drain(
            service.ask(indexed_session, QUESTION, [], "llama-x", provider, guard=guard, embedder=embedder)
        )
    expected = "Error code E3 means the device has overheated[1]. See also [9]."  # 【1】 normalised to [1]
    assert "".join(deltas) == expected == record.text
    assert record.citations == [1] and record.invalid_citations == 1
    (source,) = record.sources
    assert source.n == 1 and source.doc_name.endswith(".pdf") and source.where.startswith("p.") and source.text
    assert record.cost == "free tier" and record.cost_usd == 0.0 and record.latency_seconds >= 0
    assert record.weak_retrieval is True and record.best_dense_score is not None  # hash scores sit far below 0.75
    assert record.n_passages == 6 and record.model == "llama-x" and record.truncated is False
    # the provider saw the system prompt, the numbered passages, the question and the weak note
    (call,) = provider.calls
    assert call["system"] == SYSTEM_PROMPT and call["max_tokens"] == 700
    content = call["messages"][-1]["content"]
    assert "[6] (" in content and QUESTION in content and WEAK_RETRIEVAL_NOTE in content
    # the cost is recorded and the question counted; the log has counts only
    assert guard.read().calls == 1 and indexed_session.question_count == 1
    assert "answer: 6 passages" in caplog.text
    assert QUESTION not in caplog.text and "overheated" not in caplog.text


def test_ask_passes_history_as_plain_turns(indexed_session, guard, embedder):
    provider = FakeProvider()
    history = [("earlier question", "earlier answer"), ("", "")]
    drain(service.ask(indexed_session, QUESTION, history, "m", provider, guard=guard, embedder=embedder))
    messages = provider.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "earlier question" and messages[1]["content"] == "earlier answer"


def test_ask_requires_an_index_and_a_question(store, guard, embedder):
    session = store.create()
    with pytest.raises(ValueError, match="build the index first"):
        next(service.ask(session, QUESTION, [], "m", FakeProvider(), guard=guard, embedder=embedder))
    service.build_index(session, SAMPLE_DOCS, embedder=embedder)
    with pytest.raises(ValueError, match="type a question"):
        next(service.ask(session, "   ", [], "m", FakeProvider(), guard=guard, embedder=embedder))
    assert session.question_count == 0


def test_ask_respects_the_daily_budget(indexed_session, tmp_path, embedder):
    guard = BudgetGuard(tmp_path / "b.json", max_daily_usd=0.01)
    guard.record(0.02)
    provider = FakeProvider()
    with pytest.raises(service.BudgetExceeded, match="Demo budget reached for today"):
        next(service.ask(indexed_session, QUESTION, [], "m", provider, guard=guard, embedder=embedder))
    assert provider.calls == [] and indexed_session.question_count == 0 and guard.read().calls == 1


def test_ask_enforces_the_question_limit(tmp_path, guard, embedder):
    store = SessionStore(tmp_path / "s", settings=make_settings(max_questions_per_session=1))
    session = store.create()
    service.build_index(session, SAMPLE_DOCS, embedder=embedder)
    drain(service.ask(session, QUESTION, [], "m", FakeProvider(), guard=guard, embedder=embedder))
    provider = FakeProvider()
    with pytest.raises(LimitExceeded, match="Question limit reached: 1"):
        drain(service.ask(session, QUESTION, [], "m", provider, guard=guard, embedder=embedder))
    assert provider.calls == [] and guard.read().calls == 1


def test_preflight_rejects_missing_model_and_unpriced_paid_models(guard):
    with pytest.raises(ValueError, match="LLM_MODEL"):
        service.preflight("", FakeProvider(), guard)
    paid = FakeProvider()
    paid.name = "anthropic"
    with pytest.raises(ValueError, match="PRICES_PER_MTOK"):
        service.preflight("claude-unknown", paid, guard)
    service.preflight("claude-haiku-5-5", paid, guard)  # priced: fine
    service.preflight("llama-x", FakeProvider(), guard)  # free tier: fine


def test_ask_prices_paid_models_and_records_the_cost(indexed_session, guard, embedder):
    provider = FakeProvider()
    provider.name = "anthropic"
    _, record = drain(
        service.ask(indexed_session, QUESTION, [], "claude-haiku-5-5", provider, guard=guard, embedder=embedder)
    )
    expected = (1500 * 0.10 + 40 * 0.50) / 1e6
    assert record.cost_usd == pytest.approx(expected) and record.cost == "$0.0002"
    assert guard.read().usd == pytest.approx(expected)


def test_ask_provider_errors_propagate_and_are_not_charged(indexed_session, guard, embedder):
    provider = FakeProvider(error=ProviderError("HTTP 429 (rate_limit_exceeded): slow down"), after=1)
    gen = service.ask(indexed_session, QUESTION, [], "m", provider, guard=guard, embedder=embedder)
    assert next(gen) == "Error "  # one delta arrived before the failure
    with pytest.raises(ProviderError, match="429"):
        next(gen)
    assert guard.read().calls == 0


# --- shared embedder ----------------------------------------------------------------------


def test_shared_embedder_is_created_once_per_process():
    service.reset_shared_embedder()
    try:
        first = service.shared_embedder(make_settings())
        assert first.name == "hash-256"
        assert service.shared_embedder(make_settings(embedder="e5")) is first  # the first call's settings win
    finally:
        service.reset_shared_embedder()
