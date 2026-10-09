"""Price table, per-call cost, and a daily budget guard for the public demo.

Prices are USD per million tokens (list prices checked on 2026-10-09; see
https://platform.claude.com/docs/en/about-claude/pricing). Keep this table the single
place where prices live, and re-check it before quoting numbers in the README.
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


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> float:
    """Cost of one call. Unknown models raise so a typo cannot silently cost $0."""
    if model not in PRICES_PER_MTOK:
        raise KeyError(f"No price for model {model!r}; add it to PRICES_PER_MTOK")
    in_price, out_price = PRICES_PER_MTOK[model]
    cached = cache_read_tokens * in_price * CACHE_READ_FRACTION.get(model, 0.10)
    uncached = max(input_tokens - cache_read_tokens, 0) * in_price
    return (uncached + cached + output_tokens * out_price) / 1_000_000


@dataclass
class Spend:
    date: str
    usd: float
    calls: int


class BudgetGuard:
    """Tracks today's spend in a small JSON file; refuses calls past the daily cap."""

    def __init__(self, path: str | Path, max_daily_usd: float, clock=time.time) -> None:
        self.path = Path(path)
        self.max_daily_usd = max_daily_usd
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
        return self.read().usd + estimated_usd <= self.max_daily_usd

    def remaining_usd(self) -> float:
        return max(self.max_daily_usd - self.read().usd, 0.0)

    def record(self, usd: float) -> Spend:
        spend = self.read()
        spend = Spend(spend.date, spend.usd + usd, spend.calls + 1)
        self._write(spend)
        return spend
