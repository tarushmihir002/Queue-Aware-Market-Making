"""
Bayesian execution-aware quoter (the proposed policy).

At every decision time, for each side and each candidate quote (distance d from
the touch, size multiplier m), compute the one-period expected edge

    E[edge] = P_fill(τ | q_ahead, state) · m·s · ( E[markout_h | state, d, q] − fee )
              − γ σ² T · inventory · side · m·s                 (AS marginal inventory cost)

and quote the candidate with the highest positive edge; quote nothing on a side
where no candidate has positive edge.

* P_fill comes from the closed-form `BayesianQueueModel`: a conjugate Gamma
  posterior on the queue-depletion rate, updated online from the observed trade
  stream (`on_step_end`). Optionally Thompson-sampled for exploration.
* E[markout] comes from the offline `MarkoutModel` (gradient boosting on probe
  fills), conditioned on queue position and market state. It already includes
  the captured half-spread, so it *is* the expected gross edge per unit.
* The queue-ahead quantity for d > 0 is all volume at better prices plus the
  level at our price (`MarketState.volume_ahead`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..backtest.engine import MarketState, QuoteDecision
from ..models.features import order_features
from ..models.survival import BayesianQueueModel, CompetingRiskModel
from .base import Strategy


class BayesianQuoter(Strategy):
    name = "bayesian_queue"

    def __init__(self, cfg: dict, queue_model: BayesianQueueModel, markout_model,
                 cr_model: Optional[CompetingRiskModel] = None, dists=(0, 1, 2), size_mults=(1.0, 2.0),
                 gamma: float | None = None, horizon_s: float | None = None, thompson: bool | None = None,
                 seed: int = 0, min_edge_ticks: float = 0.0, improve: bool = True):
        super().__init__(cfg)
        sc = cfg["strategy"]
        self.qm = queue_model
        self.mm = markout_model
        self.cr = cr_model
        self.dists = list(dists)
        self.mults = list(size_mults)
        self.gamma = float(gamma if gamma is not None else sc["gamma"])
        self.T = float(horizon_s if horizon_s is not None else sc["horizon_s"])
        self.tau = float(sc.get("decision_horizon_s", 1.0))
        self.h = float(sc.get("markout_horizon_s", 1.0))
        self.thompson = bool(sc.get("bayes_thompson", True) if thompson is None else thompson)
        self.fee = float(cfg["fees"]["maker_bps"]) * 1e-4
        self.lot = float(cfg["instrument"]["lot_size"])
        self.min_edge = min_edge_ticks
        self.improve = improve
        self.rng = np.random.default_rng(seed)
        self._sig_ema = None
        self.last_diag = {}

    def reset(self):
        self._sig_ema = None

    # ---- online Bayesian update of the queue-depletion rate
    def on_step_end(self, state: MarketState, fills, trades) -> None:
        dt = self.cfg["strategy"]["requote_interval_s"]
        for side, agg in ((1, -1), (-1, 1)):
            sz = np.array([q for (_, _, q, s) in trades if s == agg])
            self.qm.update(side, len(sz), dt, sz if len(sz) else None)

    def decide(self, s: MarketState) -> QuoteDecision:
        tick = s.tick
        sig = max(s.sigma_1s, 0.3 * tick)
        self._sig_ema = sig if self._sig_ema is None else 0.9 * self._sig_ema + 0.1 * sig
        inv_cost_unit = self.gamma * self._sig_ema ** 2 * self.T * s.inventory   # per unit, signed
        dec = QuoteDecision()
        diag = {}
        for side in (1, -1):
            best = s.best_bid if side > 0 else s.best_ask
            top_same = s.bid_sz1 if side > 0 else s.ask_sz1
            lam = self.qm.sample_rate(side, self.rng) if self.thompson else None
            dists = list(self.dists)
            if self.improve and s.spread_ticks >= 3:
                dists = [-1] + dists          # one tick inside the spread, queue_ahead = 0
            # ---- build all candidate feature rows, then score the models in ONE batched call each
            rows, meta = [], []
            for d in dists:
                price = round(best - side * d * tick, 10)
                q = 0.0 if d < 0 else s.volume_ahead(side, price)
                p_fill = self.qm.prob_fill_within(side, q, self.tau, lam=lam)
                rows.append(np.concatenate([s.features, order_features(side, d, q, top_same, s.imb1, self.lot)]))
                meta.append((d, price, p_fill))
            X = np.asarray(rows, dtype=np.float32)
            e_mo_all = self.mm.predict(X, self.h)
            p_req_all = self.cr.prob_requote_within(X, self.tau) if self.cr is not None else np.zeros(len(X))
            cands = []
            for (d, price, p_fill), e_mo, p_req in zip(meta, e_mo_all, p_req_all):
                p_fill = p_fill * (1.0 - 0.5 * float(p_req))   # requote risk shortens effective exposure
                for m in self.mults:
                    qty = m * self.size
                    if abs(s.inventory + side * qty) > self.max_inv:
                        continue
                    edge = p_fill * qty * (float(e_mo) - self.fee * price) - inv_cost_unit * side * qty
                    cands.append((edge, d, qty, price, p_fill, float(e_mo)))
            if not cands:
                continue
            edge, d, qty, price, p_fill, e_mo = max(cands, key=lambda c: c[0])
            diag[side] = dict(edge=edge, dist=d, qty=qty, p_fill=p_fill, e_markout_ticks=e_mo / tick)
            if edge > self.min_edge * tick * qty:
                if side > 0:
                    dec.bid_price, dec.bid_size = price, qty
                else:
                    dec.ask_price, dec.ask_size = price, qty
        self.last_diag = diag
        return dec
