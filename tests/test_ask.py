"""End-to-end tests for scripts/ask.py with a mocked provider: no network, no API key."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from docs_chat.config import Settings
from docs_chat.costs import BudgetGuard, cost_usd
from docs_chat.llm import MAX_TOKENS, SYSTEM_PROMPT, WEAK_RETRIEVAL_NOTE
from docs_chat.providers import LLMResult, ProviderError

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What does error code E3 mean?"
ANSWER = "Error code E3 means the device has overheated: unplug it and let it cool for 30 minutes [1]. See also [9]."


def load_script(name: str):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(f"docs_chat_scripts_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory) -> Path:
    """The sample docs indexed with the offline hash embedder, built once per module."""
    out = tmp_path_factory.mktemp("index")
    ingest = load_script("ingest")
    assert ingest.main([str(ROOT / "eval" / "sample_docs"), "--out", str(out), "--embedder", "hash"]) == 0
    return out


@pytest.fixture(scope="module")
def ask():
    return load_script("ask")


@pytest.fixture
def guard(tmp_path) -> BudgetGuard:
    return BudgetGuard(tmp_path / "budget.json", max_daily_usd=2.0)


class FakeProvider:
    """LLMProvider stand-in: streams a fixed answer piece by piece and records the call."""

    def __init__(self, name: str, text: str = ANSWER, input_tokens: int = 3500, output_tokens: int = 300):
        self.name, self.text = name, text
        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        self.calls = []

    def _record(self, messages, system, model, max_tokens) -> LLMResult:
        self.calls.append({"messages": list(messages), "system": system, "model": model, "max_tokens": max_tokens})
        return LLMResult(
            text=self.text, input_tokens=self.input_tokens, output_tokens=self.output_tokens, cache_read_tokens=0,
            model=model, elapsed_seconds=0.05, usage_estimated=False, stop_reason="stop",
        )

    def stream(self, messages, system, model, max_tokens):
        result = self._record(messages, system, model, max_tokens)
        yield from re.split(r"(?<= )", self.text)  # word-sized deltas that re-join to the exact text
        return result

    def complete(self, messages, system, model, max_tokens):
        return self._record(messages, system, model, max_tokens)


PREVIEW_LIMIT = 200 + 1  # 200 characters plus the ellipsis


def base_args(index_dir: Path, model: str) -> list[str]:
    return [QUESTION, "--index", str(index_dir), "--embedder", "hash", "--model", model]


@pytest.mark.parametrize(
    ("provider_name", "model", "expected_cost"),
    [("openai_compat", "llama-3.3-70b-versatile", "free tier"), ("anthropic", "claude-haiku-5-5", "$0.0005")],
)
def test_ask_end_to_end_with_mocked_provider(ask, index_dir, guard, capsys, provider_name, model, expected_cost):
    provider = FakeProvider(provider_name)
    assert ask.main(base_args(index_dir, model), provider=provider, guard=guard) == 0
    out = capsys.readouterr().out
    lines = out.rstrip("\n").split("\n")

    # loading is reported first and is not part of the answer latency
    assert re.match(r"loaded index \(\d+ chunks\) \+ embedder hash-256 in \d+\.\d s$", lines[0])
    # the answer was streamed to stdout, after a note that the hash embedder's scores are below the threshold
    assert "note: retrieval was weak" in out
    assert ANSWER in out

    # cited passages: [1] exists, [9] does not; preview = doc, page, first 200 characters
    assert "Sources:" in lines
    source_lines = [line for line in lines if line.startswith("  [")]
    assert len(source_lines) == 1
    assert source_lines[0].startswith("  [1] ") and ".pdf p." in source_lines[0]
    assert len(source_lines[0].split(": ", 1)[1]) <= PREVIEW_LIMIT
    assert "invalid citations dropped: 1" in lines

    # final line
    assert re.fullmatch(rf"cost {re.escape(expected_cost)} · \d+\.\d s · 6 passages", lines[-1])

    # the provider got the system prompt, the numbered passages + question + weak-retrieval note, max_tokens=700
    (call,) = provider.calls
    assert call["system"] == SYSTEM_PROMPT and call["model"] == model and call["max_tokens"] == MAX_TOKENS
    assert [m["role"] for m in call["messages"]] == ["user"]
    content = call["messages"][0]["content"]
    assert "[1] (" in content and "[6] (" in content and QUESTION in content and WEAK_RETRIEVAL_NOTE in content

    # every call is recorded: free-tier calls as $0, priced ones at their cost
    spend = guard.read()
    assert spend.calls == 1
    assert spend.usd == pytest.approx(0.0 if provider_name == "openai_compat" else cost_usd(model, 3500, 300))



def test_ask_refuses_when_the_daily_budget_is_reached(ask, index_dir, tmp_path, capsys):
    guard = BudgetGuard(tmp_path / "budget.json", max_daily_usd=0.01)
    guard.record(0.02)
    provider = FakeProvider("anthropic")
    assert ask.main(base_args(index_dir, "claude-haiku-5-5"), provider=provider, guard=guard) == ask.EXIT_BUDGET
    captured = capsys.readouterr()
    assert "Demo budget reached for today" in captured.err
    assert provider.calls == []  # the API was not called
    assert guard.read().calls == 1  # and nothing new was recorded
    assert "cost" not in captured.out


def test_ask_exits_2_when_the_provider_is_not_configured(ask, index_dir, guard, capsys, monkeypatch):
    monkeypatch.setattr(ask, "settings", Settings(llm_provider="openai_compat", llm_api_key="", llm_model=""))
    assert ask.main([QUESTION, "--index", str(index_dir), "--embedder", "hash"], guard=guard) == 2
    assert "LLM_API_KEY is not set" in capsys.readouterr().err

    # a configured provider but no model is the same kind of error
    assert ask.main(base_args(index_dir, ""), provider=FakeProvider("openai_compat"), guard=guard) == 2
    assert "LLM_MODEL" in capsys.readouterr().err

    # a paid provider with a model missing from the price table is refused before any call
    provider = FakeProvider("anthropic")
    assert ask.main(base_args(index_dir, "claude-unknown"), provider=provider, guard=guard) == 2
    assert "PRICES_PER_MTOK" in capsys.readouterr().err
    assert provider.calls == [] and guard.read().calls == 0


def test_ask_retrieval_only_needs_no_provider_or_budget(ask, index_dir, capsys):
    assert ask.main([QUESTION, "--index", str(index_dir), "--embedder", "hash", "--retrieval-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert re.match(r"loaded index \(\d+ chunks\) \+ embedder hash-256 in \d+\.\d s$", lines[0])
    assert lines[1].startswith("retrieval: 6 passages")
    out = "\n".join(lines)
    assert "[1] " in out and "bm25=" in out
    assert "cost" not in out and "Sources" not in out


class FailingProvider(FakeProvider):
    """Streams ``after_deltas`` pieces of the answer, then raises the given error (like a dropped connection)."""

    def __init__(self, name: str, error: Exception, after_deltas: int = 0):
        super().__init__(name)
        self.error, self.after_deltas = error, after_deltas

    def stream(self, messages, system, model, max_tokens):
        self._record(messages, system, model, max_tokens)
        yield from re.split(r"(?<= )", self.text)[: self.after_deltas]
        raise self.error


@pytest.mark.parametrize("after_deltas", [0, 2], ids=["before any text", "half-way through the answer"])
def test_ask_reports_api_errors_in_one_line_with_exit_4(ask, index_dir, guard, capsys, after_deltas):
    error = ProviderError("HTTP 404 (model_not_found): The model `llama-x` does not exist or you do not have access to it.")
    provider = FailingProvider("openai_compat", error, after_deltas=after_deltas)
    assert ask.main(base_args(index_dir, "llama-x"), provider=provider, guard=guard) == ask.EXIT_API_ERROR == 4
    captured = capsys.readouterr()
    assert captured.err.rstrip().splitlines()[-1] == f"error: {error}"
    assert "Traceback" not in captured.err
    assert "cost" not in captured.out and "Sources" not in captured.out
    assert guard.read().calls == 0  # a failed call is not charged to the budget
