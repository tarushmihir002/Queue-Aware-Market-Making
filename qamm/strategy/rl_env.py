"""
Reinforcement-learning environment for queue-aware market making.

Observation (float32):
    13 market features (qamm.models.features.FEATURE_NAMES)
  + inventory / max_inv, (inventory / max_inv)², t_frac
  + our resting bid: queue_ratio (−1 if none), our resting ask: queue_ratio (−1 if none)
  + last-step signed fill flow

Action (discrete, 32):  (bid_dist, ask_dist, size_mult)
    bid_dist, ask_dist ∈ {skip, 0, 1, 2} ticks from the touch
    size_mult         ∈ {1, 2} × quote_size

Reward:  Δ mark-to-mid P&L (fees included in cash) − penalty · inv² · dt,
         scaled by 1/(tick · quote_size) so one unit ≈ one tick on one lot.

The env wraps the same BacktestEngine machinery (ReplayExecution on recorded
data by default), so RL results are directly comparable with the other
strategies. Gymnasium-compatible if gymnasium is installed.
"""
from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np

from ..backtest.engine import BacktestEngine, QuoteDecision
from ..backtest.execution import ReplayExecution
from ..models.features import FEATURE_NAMES, compute_features

try:
    import gymnasium as gym
    _Base = gym.Env
except Exception:  # pragma: no cover
    gym = None
    _Base = object

DIST_CHOICES = [None, 0, 1, 2]
SIZE_MULTS = [1.0, 2.0]
ACTION_TABLE = list(product(DIST_CHOICES, DIST_CHOICES, SIZE_MULTS))   # 32 actions
OBS_DIM = len(FEATURE_NAMES) + 6


class _ActionStrategy:
    """Strategy shim: the env sets `.decision` before each engine step."""
    name = "rl"

    def __init__(self):
        self.decision = QuoteDecision()

    def reset(self): pass
    def decide(self, s): return self.decision
    def on_fill(self, f, s): pass
    def on_step_end(self, s, fills, trades): pass


class MarketMakingEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, datasets: list, features: Optional[list] = None, episode_len_steps: int = 1200,
                 seed: int = 0, random_start: bool = True):
        self.cfg = cfg
        self.datasets = datasets
        self.features = features if features is not None else [compute_features(md) for md in datasets]
        self.L = int(episode_len_steps)
        self.rng = np.random.default_rng(seed)
        self.random_start = random_start
        self.dt = float(cfg["strategy"]["requote_interval_s"])
        self.size = float(cfg["strategy"]["quote_size"])
        self.max_inv = float(cfg["strategy"]["max_inventory"])
        self.penalty = float(cfg["strategy"]["inventory_penalty"])
        self.tick = float(datasets[0].tick_size)
        self.scale = 1.0 / (self.tick * self.size)
        if gym is not None:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
            self.action_space = gym.spaces.Discrete(len(ACTION_TABLE))
        self._shim = _ActionStrategy()
        self.engine: Optional[BacktestEngine] = None

    # ------------------------------------------------------------------
    def _obs(self):
        s = self.state
        bq = aq = -1.0
        for o in s.open_orders.values():
            top = s.bid_sz1 if o.side > 0 else s.ask_sz1
            r = o.queue_ahead / max(top, 1e-9)
            if o.side > 0: bq = r
            else: aq = r
        inv = s.inventory / self.max_inv
        return np.concatenate([s.features, [inv, inv * inv, s.t_frac, bq, aq, self._last_flow]]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        k = int(self.rng.integers(len(self.datasets)))
        md, F = self.datasets[k], self.features[k]
        total = md.book.ts[-1] - md.book.ts[0]
        need = self.L * self.dt + 5.0
        start = float(self.rng.uniform(1.0, max(1.0, total - need))) if self.random_start else 1.0
        ex = ReplayExecution(md, start_ts=start)
        self.engine = BacktestEngine(self.cfg, ex, self._shim, features=F, dt=self.dt,
                                     max_duration_s=self.L * self.dt)
        self.engine._n_bid = self.engine._n_ask = 0
        self.engine._row_trade_ptr = 0
        self._shim.reset()
        self.inv = self.cash = 0.0
        self.t0 = ex.ts
        self.step_i = 0
        self._last_flow = 0.0
        self.state = self.engine._state(self.inv, self.cash, 0.0, self.L * self.dt)
        self._prev_pnl = 0.0
        self.ep_fills = 0
        self.ep_fees = 0.0
        return self._obs(), {}

    def step(self, action: int):
        bd, ad, m = ACTION_TABLE[int(action)]
        s = self.state
        dec = QuoteDecision()
        if bd is not None:
            dec.bid_price, dec.bid_size = s.best_bid - bd * self.tick, m * self.size
        if ad is not None:
            dec.ask_price, dec.ask_size = s.best_ask + ad * self.tick, m * self.size
        self._shim.decision = dec
        eng, ex = self.engine, self.engine.ex
        eng._reconcile(dec, self.inv)
        fills, ended = ex.advance(self.dt)
        flow = 0.0
        for f in fills:
            fee = eng.maker_fee * f.price * f.qty
            self.inv += f.side * f.qty
            self.cash -= f.side * f.qty * f.price + fee
            self.ep_fees += fee
            flow += f.side * f.qty
            self.ep_fills += 1
        self._last_flow = flow / self.size
        self.step_i += 1
        elapsed = ex.ts - self.t0
        self.state = eng._state(self.inv, self.cash, elapsed, self.L * self.dt)
        pnl = self.state.mtm_pnl
        reward = (pnl - self._prev_pnl - self.penalty * self.inv ** 2 * self.dt) * self.scale
        self._prev_pnl = pnl
        terminated = ended
        truncated = self.step_i >= self.L
        if terminated or truncated:
            # charge liquidation at the touch so the agent learns to flatten
            px = ex.best_bid if self.inv > 0 else ex.best_ask
            liq = abs(self.inv) * (abs(px - ex.mid) + eng.taker_fee * px)
            reward -= liq * self.scale
            for oid in list(ex.open_orders()):
                ex.cancel(oid)
        return self._obs(), float(reward), bool(terminated), bool(truncated), {"pnl": pnl, "inv": self.inv,
                                                                                "fills": self.ep_fills}
