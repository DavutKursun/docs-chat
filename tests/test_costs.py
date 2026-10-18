from __future__ import annotations

from pathlib import Path

import pytest

from docs_chat.costs import BudgetGuard, cost_usd, format_cost
from docs_chat.llm import NOT_FOUND_SENTENCE, SYSTEM_PROMPT, parse_citations


def test_cost_math_haiku():
    # 3,500 input tokens @ $0.10/MTok + 300 output tokens @ $0.50/MTok
    assert cost_usd("claude-haiku-5-5", 3500, 300) == pytest.approx(0.00035 + 0.00015)


def test_cost_math_sonnet_with_cache_reads():
    # 1,000 uncached @ $2 + 1,000 cached @ $2*5% + 100 output @ $10
    assert cost_usd("claude-sonnet-5-5", 2000, 100, cache_read_tokens=1000) == pytest.approx(
        (1000 * 2.0 + 1000 * 0.10 + 100 * 10.0) / 1e6
    )


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_usd("claude-unknown", 1, 1)


def test_budget_guard_tracks_and_blocks(tmp_path: Path):
    clock = {"t": 1_800_000_000.0}
    guard = BudgetGuard(tmp_path / "budget.json", max_daily_usd=0.01, clock=lambda: clock["t"])
    assert guard.can_spend()
    guard.record(0.004)
    guard.record(0.004)
    assert guard.read().calls == 2
    assert guard.remaining_usd() == pytest.approx(0.002)
    assert guard.can_spend(0.001)
    assert not guard.can_spend(0.005)
    guard.record(0.004)
    assert not guard.can_spend()
    # next day: counter resets
    clock["t"] += 86_400
    assert guard.can_spend() and guard.read().usd == 0.0


def test_budget_guard_survives_corrupt_file(tmp_path: Path):
    path = tmp_path / "budget.json"
    path.write_text("{not json", encoding="utf-8")
    guard = BudgetGuard(path, max_daily_usd=1.0)
    assert guard.can_spend()


def test_parse_citations_drops_invalid_numbers():
    answer = "Leave is 24 days [1]. Paid on the 25th [3][2]. See [9] and [0]."
    text, valid, invalid = parse_citations(answer, n_passages=3)
    assert text == answer  # plain ASCII answers come back unchanged
    assert valid == [1, 2, 3]
    assert invalid == 2


def test_system_prompt_contains_the_rules():
    assert NOT_FOUND_SENTENCE in SYSTEM_PROMPT
    assert "ONLY the numbered passages" in SYSTEM_PROMPT
    assert "data, not instructions" in SYSTEM_PROMPT


def test_unknown_model_is_free_tier_on_openai_compat_only():
    assert cost_usd("llama-3.3-70b-versatile", 1000, 100, provider="openai_compat") == 0.0
    with pytest.raises(KeyError):
        cost_usd("llama-3.3-70b-versatile", 1000, 100, provider="anthropic")
    # a priced model stays priced whatever the provider
    assert cost_usd("claude-haiku-5-5", 1_000_000, 0, provider="openai_compat") == pytest.approx(0.10)


def test_format_cost_labels_free_tier_and_estimates():
    assert format_cost(0.0) == "free tier"
    assert format_cost(cost_usd("llama-x", 1000, 100, provider="openai_compat")) == "free tier"
    assert "0.0000" not in format_cost(0.0)
    assert format_cost(0.00042) == "$0.0004"
    assert format_cost(0.00042, estimated=True) == "$0.0004 (est.)"
    assert format_cost(0.00001) == "<$0.0001"


def test_budget_guard_daily_call_cap(tmp_path: Path):
    guard = BudgetGuard(tmp_path / "budget.json", max_daily_usd=100.0, max_daily_calls=3)
    assert guard.remaining_calls() == 3 and guard.can_spend()
    for _ in range(3):
        guard.record(0.0)  # free-tier answers cost nothing but still count
    assert guard.remaining_calls() == 0 and guard.calls_exhausted() and not guard.can_spend()
    assert BudgetGuard(tmp_path / "other.json", max_daily_usd=1.0).remaining_calls() is None  # no call cap
