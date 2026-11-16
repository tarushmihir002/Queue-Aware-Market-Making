"""Strategy interface and two naive baselines."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..backtest.engine import MarketState, QuoteDecision


class Strategy(ABC):
    name = "strategy"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.size = float(cfg["strategy"]["quote_size"])
        self.max_inv = float(cfg["strategy"]["max_inventory"])

    def reset(self) -> None:
        pass

    @abstractmethod
    def decide(self, state: MarketState) -> QuoteDecision: ...

    def on_fill(self, fill, state: MarketState) -> None:
        pass

    def on_step_end(self, state: MarketState, fills, trades) -> None:
        pass


class JoinTouchStrategy(Strategy):
    """Baseline 0: always join the best bid and best ask with fixed size (no skew)."""
    name = "join_touch"

    def decide(self, s: MarketState) -> QuoteDecision:
        return QuoteDecision(s.best_bid, self.size, s.best_ask, self.size)


class FixedSpreadStrategy(Strategy):
    """Baseline 1: symmetric quotes `half_spread_ticks` from mid with linear inventory skew."""
    name = "fixed_spread"

    def __init__(self, cfg: dict, half_spread_ticks: float = 1.0, skew_ticks_per_unit: float = 0.05):
        super().__init__(cfg)
        self.h = half_spread_ticks
        self.skew = skew_ticks_per_unit
        self.name = f"fixed_spread_{half_spread_ticks:g}t"

    def decide(self, s: MarketState) -> QuoteDecision:
        r = s.mid - self.skew * s.inventory * s.tick
        return QuoteDecision(r - self.h * s.tick, self.size, r + self.h * s.tick, self.size)
