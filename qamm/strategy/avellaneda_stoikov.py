"""
Avellaneda–Stoikov (2008) market maker — the classical baseline.

    reservation price   r = s − q · γ · σ² · (T − t)
    optimal spread      δ = γ σ² (T − t) + (2/γ) · ln(1 + γ/κ)
    quotes              bid = r − δ/2,  ask = r + δ/2

σ (per-second mid volatility) is estimated online from the realised-vol
feature; κ is the decay of the fill intensity with distance, λ(δ) = A e^{−κδ},
calibrated from the probe dataset with `calibrate_kappa`. (T − t) is a rolling
pseudo-horizon (config `horizon_s`) since the session is not terminal.
"""
from __future__ import annotations

import numpy as np

from ..backtest.engine import MarketState, QuoteDecision
from .base import Strategy


class AvellanedaStoikov(Strategy):
    name = "avellaneda_stoikov"

    def __init__(self, cfg: dict, gamma: float | None = None, kappa: float | None = None,
                 horizon_s: float | None = None, sigma_floor_ticks: float = 0.3, size: float | None = None):
        super().__init__(cfg)
        sc = cfg["strategy"]
        self.gamma = float(gamma if gamma is not None else sc["gamma"])
        self.kappa_tick = float(kappa if kappa is not None else sc["kappa"])  # per tick
        self.T = float(horizon_s if horizon_s is not None else sc["horizon_s"])
        self.sigma_floor_ticks = sigma_floor_ticks
        if size is not None:
            self.size = size
        self._sig_ema = None

    def reset(self):
        self._sig_ema = None

    def decide(self, s: MarketState) -> QuoteDecision:
        tick = s.tick
        sig = max(s.sigma_1s, self.sigma_floor_ticks * tick)
        self._sig_ema = sig if self._sig_ema is None else 0.9 * self._sig_ema + 0.1 * sig
        sig2 = self._sig_ema ** 2
        kappa = self.kappa_tick / tick                      # per price unit
        r = s.mid - s.inventory * self.gamma * sig2 * self.T
        delta = self.gamma * sig2 * self.T + (2.0 / self.gamma) * np.log1p(self.gamma / kappa)
        half = max(delta / 2.0, 0.5 * tick)
        bid = r - half
        ask = r + half
        # never quote worse than one tick inside the opposite touch is handled by the engine;
        # here we only make sure we are not more aggressive than the current touch.
        bid = min(bid, s.best_bid + (s.spread_ticks - 1) * tick) if s.spread_ticks > 1 else min(bid, s.best_bid)
        ask = max(ask, s.best_ask - (s.spread_ticks - 1) * tick) if s.spread_ticks > 1 else max(ask, s.best_ask)
        return QuoteDecision(bid, self.size, ask, self.size)


def calibrate_kappa(ds, tau: float = 1.0) -> float:
    """Fit λ(δ) = A exp(−κ δ) to the empirical fill rate within τ by distance (ticks).

    Returns κ per tick. Uses the probes that joined the back of the queue."""
    from ..models.survival import CAUSE_FILL
    rates, dists = [], []
    for d in np.unique(ds.dist):
        m = (ds.dist == d) & (ds.queue_ratio >= 0.99)
        if m.sum() < 50:
            continue
        p = np.mean((ds.cause[m] == CAUSE_FILL) & (ds.time[m] <= tau))
        if p > 0:
            rates.append(np.log(p)); dists.append(float(d))
    if len(dists) < 2:
        return 1.5
    slope = np.polyfit(dists, rates, 1)[0]
    return float(max(-slope, 0.05))
