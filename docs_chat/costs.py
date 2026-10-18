"""Price table, per-call cost, and a daily budget guard for the public demo.

Prices are USD per million tokens (list prices checked on 2026-10-09; see
https://platform.claude.com/docs/en/about-claude/pricing). Keep this table the single
place where prices live, and re-check it before quoting numbers in the README.

Only Anthropic models are priced. On an OpenAI-compatible provider a model that is not
in the table is treated as a free tier: it costs $0.00 and ``format_cost`` shows
"free tier" rather than "$0.0000", so nobody mistakes it for a measured cost. Add a
row to ``PRICES_PER_MTOK`` (keyed by the provider's model id) to price a paid model.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # model id: (input $/MTok, output $/MTok)
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
CACHE_READ_FRACTION = {"claude-haiku-5-5": 0.10, "claude-sonnet-5-5": 0.05}  # of input price


def cost_usd(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0, provider: str = "anthropic"
) -> float:
    """Cost of one call; ``input_tokens`` is the total prompt, ``cache_read_tokens`` its cached subset.

    Unknown models raise so a typo cannot silently cost $0 — except on an OpenAI-compatible
    provider, where an unpriced model is a free tier and costs 0.0 (see the module docstring).
    """
    if model not in PRICES_PER_MTOK:
        if provider == "openai_compat":
            return 0.0
        raise KeyError(f"No price for model {model!r}; add it to PRICES_PER_MTOK")
    in_price, out_price = PRICES_PER_MTOK[model]
    cached = cache_read_tokens * in_price * CACHE_READ_FRACTION.get(model, 0.10)
    uncached = max(input_tokens - cache_read_tokens, 0) * in_price
    return (uncached + cached + output_tokens * out_price) / 1_000_000


def format_cost(usd: float, estimated: bool = False) -> str:
    """Cost for the UI/CLI: "$0.0004", "<$0.0001", "$0.0004 (est.)" or "free tier".

    A cost of exactly zero only happens for unpriced free-tier models, so it is labelled
    "free tier" instead of "$0.0000" to avoid looking like a measured cost.
    """
    if usd <= 0:
        return "free tier"
    text = "<$0.0001" if usd < 0.0001 else f"${usd:.4f}"
    return f"{text} (est.)" if estimated else text


@dataclass
class Spend:
    date: str
    usd: float
    calls: int


class BudgetGuard:
    """Tracks today's spend and call count in a small JSON file; refuses calls past either daily cap.

    ``max_daily_calls`` caps the number of answers per UTC day for the whole deployment (free-tier
    demos record $0 per call, so the dollar cap alone would never trip); None = no call cap.
    """

    def __init__(
        self, path: str | Path, max_daily_usd: float, max_daily_calls: int | None = None, clock=time.time
    ) -> None:
        self.path = Path(path)
        self.max_daily_usd = max_daily_usd
        self.max_daily_calls = max_daily_calls
        self._clock = clock

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self._clock()))

    def read(self) -> Spend:
        if not self.path.exists():
            return Spend(self._today(), 0.0, 0)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Spend(self._today(), 0.0, 0)
        if data.get("date") != self._today():
            return Spend(self._today(), 0.0, 0)
        return Spend(data["date"], float(data.get("usd", 0.0)), int(data.get("calls", 0)))

    def _write(self, spend: Spend) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".budget-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"date": spend.date, "usd": spend.usd, "calls": spend.calls}, fh)
        os.replace(tmp, self.path)

    def can_spend(self, estimated_usd: float = 0.0) -> bool:
        spend = self.read()
        if self.max_daily_calls is not None and spend.calls >= self.max_daily_calls:
            return False
        return spend.usd + estimated_usd <= self.max_daily_usd

    def calls_exhausted(self) -> bool:
        return self.max_daily_calls is not None and self.read().calls >= self.max_daily_calls

    def remaining_usd(self) -> float:
        return max(self.max_daily_usd - self.read().usd, 0.0)

    def remaining_calls(self) -> int | None:
        return None if self.max_daily_calls is None else max(self.max_daily_calls - self.read().calls, 0)

    def record(self, usd: float) -> Spend:
        spend = self.read()
        spend = Spend(spend.date, spend.usd + usd, spend.calls + 1)
        self._write(spend)
        return spend
