"""
Event-driven backtesting engine for passive market-making strategies.

Loop (decision cadence `dt`):
    state  = market + inventory snapshot
    quotes = strategy.decide(state)            -> bid/ask price & size (or None)
    reconcile resting orders (keep an order whose price is unchanged, to
    preserve queue priority; cancel/replace otherwise)
    fills  = execution.advance(dt)
    update cash / inventory / fees; notify strategy

P&L accounting (all in quote currency):
    mtm_pnl(t)      = cash(t) + inventory(t) · mid(t)
    spread_revenue  = Σ_fills side · qty · (mid_at_fill − price)        (captured half-spread)
    fees            = Σ_fills maker_bps · 1e-4 · price · qty
    inventory_pnl   = mtm_pnl − spread_revenue + fees                     (exact residual: price moves
                                                                           while holding inventory)
    adverse_sel_h   = −Σ_fills side · qty · (mid(t_fill + h) − mid(t_fill))  (markout-based toxicity)
    inventory_cost  = penalty · Σ inv² · dt                               (risk charge in the objective)
    liquidation     = cost of unwinding terminal inventory at the touch + taker fee

Objective reported: spread_revenue − adverse_sel_h − fees − inventory_cost.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..models.features import FEATURE_NAMES, OnlineFeatures, compute_features
from .execution import ClosedLoopExecution, ExecutionModel, Fill, ReplayExecution


@dataclass
class QuoteDecision:
    bid_price: Optional[float] = None
    bid_size: float = 0.0
    ask_price: Optional[float] = None
    ask_size: float = 0.0


@dataclass
class MarketState:
    ts: float
    mid: float
    best_bid: float
    best_ask: float
    bid_sz1: float
    ask_sz1: float
    tick: float
    features: np.ndarray            # FEATURE_NAMES vector
    inventory: float
    cash: float
    mtm_pnl: float
    t_frac: float                   # elapsed / horizon ∈ [0, 1]
    elapsed: float
    open_orders: Dict[int, object]
    last_trades: list = field(default_factory=list)
    exec: Optional[ExecutionModel] = None

    @property
    def spread_ticks(self) -> float:
        return (self.best_ask - self.best_bid) / self.tick

    @property
    def imb1(self) -> float:
        return float(self.features[1])

    @property
    def sigma_1s(self) -> float:
        """Per-second mid volatility in price units (from the 10 s realised vol feature)."""
        return float(self.features[9]) * self.tick / np.sqrt(10.0)

    def level_size(self, side: int, price: float) -> float:
        return self.exec.level_size(side, price) if self.exec is not None else 0.0

    def volume_ahead(self, side: int, price: float) -> float:
        """All resting volume that must trade before an order at `price` on `side` does
        (levels strictly better than ours + the level at our price)."""
        ex = self.exec
        if ex is None:
            return 0.0
        tot = 0.0
        best = self.best_bid if side > 0 else self.best_ask
        k = 0
        p = best
        while (side > 0 and p > price + 1e-9) or (side < 0 and p < price - 1e-9):
            tot += ex.level_size(side, p)
            p = round(p - side * self.tick, 10)
            k += 1
            if k > 50:
                break
        return tot + ex.level_size(side, price)


@dataclass
class BacktestResult:
    name: str
    ts: np.ndarray
    mid: np.ndarray
    inventory: np.ndarray
    cash: np.ndarray
    mtm_pnl: np.ndarray
    n_bid_quotes: int
    n_ask_quotes: int
    fills: np.ndarray                # structured: ts, side, price, qty, queue_at_place, placed_ts, mid_at_fill
    markouts: np.ndarray             # (n_fills, H) side·(mid(t+h) − mid(t))
    markout_horizons: np.ndarray
    fees_paid: float
    inventory_cost: float
    liquidation_cost: float
    dt: float
    wall_time_s: float
    extra: dict = field(default_factory=dict)

    @property
    def n_fills(self) -> int:
        return len(self.fills)


FILL_DTYPE = np.dtype([("ts", "f8"), ("side", "i1"), ("price", "f8"), ("qty", "f8"),
                       ("queue_at_place", "f8"), ("placed_ts", "f8"), ("mid_at_fill", "f8"),
                       ("fee", "f8")])


class BacktestEngine:
    def __init__(self, cfg: dict, execution: ExecutionModel, strategy, features: Optional[np.ndarray] = None,
                 dt: Optional[float] = None, max_duration_s: Optional[float] = None, verbose: bool = False):
        self.cfg = cfg
        self.ex = execution
        self.strategy = strategy
        self.dt = float(dt or cfg["strategy"]["requote_interval_s"])
        self.max_duration = max_duration_s
        self.verbose = verbose
        self.maker_fee = float(cfg["fees"]["maker_bps"]) * 1e-4
        self.taker_fee = float(cfg["fees"]["taker_bps"]) * 1e-4
        self.max_inv = float(cfg["strategy"]["max_inventory"])
        self.inv_penalty = float(cfg["strategy"]["inventory_penalty"])
        self.horizons = np.asarray(cfg["strategy"]["markout_horizons_s"], dtype=float)
        self.tick = execution.tick
        self._closed = isinstance(execution, ClosedLoopExecution)
        if self._closed:
            self._online = OnlineFeatures(self.tick)
            self._feat = None
        else:
            self._feat = features if features is not None else compute_features(execution.md)
            self._online = None

    # ------------------------------------------------------------------
    def _features_now(self) -> np.ndarray:
        if self._closed:
            s = self.ex.sim
            i0, i1 = self.ex.new_rows()
            # feed new rows; trades keyed to the row's ts
            tp = self._row_trade_ptr
            for i in range(i0, i1):
                ts = s.rec_ts[i]
                new_tr = []
                while tp < len(s.rec_trades) and s.rec_trades[tp][0] <= ts:
                    t, p, q, sd = s.rec_trades[tp]
                    new_tr.append((t, sd, q))
                    tp += 1
                bs3 = s.rec_bs[i][:3].sum(); as3 = s.rec_as[i][:3].sum()
                self._online.update(ts, s.rec_bp[i][0], s.rec_bs[i][0], s.rec_ap[i][0], s.rec_as[i][0],
                                    (bs3 - as3) / max(bs3 + as3, 1e-12), new_tr)
            self._row_trade_ptr = tp
            if self._online.cur is None:
                return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            return self._online.vector()
        return self._feat[self.ex.feature_row()]

    def _state(self, inv, cash, elapsed, horizon) -> MarketState:
        ex = self.ex
        bs, asz = ex.top_sizes()
        mid = ex.mid
        return MarketState(ex.ts, mid, ex.best_bid, ex.best_ask, bs, asz, self.tick, self._features_now(),
                           inv, cash, cash + inv * mid, min(elapsed / horizon, 1.0), elapsed,
                           ex.open_orders(), getattr(ex, "last_trades", []), ex)

    def _reconcile(self, dec: QuoteDecision, inv: float) -> None:
        ex = self.ex
        want = {}
        if dec.bid_price is not None and dec.bid_size > 0 and inv + dec.bid_size <= self.max_inv + 1e-9:
            p = min(ex.round_px(dec.bid_price) if hasattr(ex, "round_px") else round(round(dec.bid_price / self.tick) * self.tick, 10),
                    round(ex.best_ask - self.tick, 10))
            want[1] = (p, dec.bid_size)
        if dec.ask_price is not None and dec.ask_size > 0 and inv - dec.ask_size >= -self.max_inv - 1e-9:
            p = max(round(round(dec.ask_price / self.tick) * self.tick, 10), round(ex.best_bid + self.tick, 10))
            want[-1] = (p, dec.ask_size)
        # cancel what we no longer want; keep same-price orders (queue priority)
        for oid, o in list(ex.open_orders().items()):
            w = want.get(o.side)
            if w is None or abs(w[0] - o.price) > 1e-9 or o.remaining < 0.5 * w[1]:
                ex.cancel(oid)
            else:
                want.pop(o.side)
        for side, (p, s) in want.items():
            try:
                ex.place(side, p, s)
                if side > 0:
                    self._n_bid += 1
                else:
                    self._n_ask += 1
            except ValueError:
                pass

    # ------------------------------------------------------------------
    def run(self) -> BacktestResult:
        t_wall = _time.perf_counter()
        ex, st = self.ex, self.strategy
        st.reset()
        self._n_bid = self._n_ask = 0
        self._row_trade_ptr = 0
        inv = cash = 0.0
        fees = inv_cost = 0.0
        t0 = ex.ts
        horizon = self.max_duration or (ex.end_ts - t0)
        rec_ts, rec_mid, rec_inv, rec_cash, rec_pnl = [], [], [], [], []
        fills_out: List[tuple] = []
        step = 0
        while True:
            elapsed = ex.ts - t0
            state = self._state(inv, cash, elapsed, horizon)
            dec = st.decide(state)
            self._reconcile(dec, inv)
            fills, ended = ex.advance(self.dt)
            mid_after = ex.mid
            for f in fills:
                fee = self.maker_fee * f.price * f.qty
                inv += f.side * f.qty
                cash -= f.side * f.qty * f.price + fee
                fees += fee
                fills_out.append((f.ts, f.side, f.price, f.qty, f.queue_at_place, f.placed_ts, mid_after, fee))
                st.on_fill(f, state)
            inv_cost += self.inv_penalty * inv * inv * self.dt
            st.on_step_end(state, fills, getattr(ex, "last_trades", []))
            rec_ts.append(ex.ts); rec_mid.append(mid_after); rec_inv.append(inv); rec_cash.append(cash)
            rec_pnl.append(cash + inv * mid_after)
            step += 1
            if self.verbose and step % 2000 == 0:
                print(f"  t={ex.ts - t0:8.1f}s inv={inv:+.2f} pnl={cash + inv * mid_after:+.4f} fills={len(fills_out)}")
            if ended or (self.max_duration and elapsed + self.dt >= self.max_duration):
                break
        # terminal liquidation at the touch with taker fee
        ex.cancel_all() if hasattr(ex, "cancel_all") else [ex.cancel(o) for o in list(ex.open_orders())]
        liq = 0.0
        if abs(inv) > 1e-9:
            px = ex.best_bid if inv > 0 else ex.best_ask
            liq = abs(inv) * (abs(px - ex.mid) + self.taker_fee * px)
        fills_arr = np.array(fills_out, dtype=FILL_DTYPE) if fills_out else np.empty(0, dtype=FILL_DTYPE)
        markouts = self._markouts(fills_arr)
        return BacktestResult(
            name=getattr(st, "name", type(st).__name__), ts=np.asarray(rec_ts), mid=np.asarray(rec_mid),
            inventory=np.asarray(rec_inv), cash=np.asarray(rec_cash), mtm_pnl=np.asarray(rec_pnl),
            n_bid_quotes=self._n_bid, n_ask_quotes=self._n_ask, fills=fills_arr, markouts=markouts,
            markout_horizons=self.horizons, fees_paid=fees, inventory_cost=inv_cost, liquidation_cost=liq,
            dt=self.dt, wall_time_s=_time.perf_counter() - t_wall,
        )

    def _book_path(self):
        if self._closed:
            s = self.ex.sim
            ts = np.asarray(s.rec_ts)
            mid = 0.5 * (np.array([r[0] for r in s.rec_bp]) + np.array([r[0] for r in s.rec_ap]))
            return ts, mid
        b = self.ex.md.book
        return b.ts, b.mid

    def _markouts(self, fills: np.ndarray) -> np.ndarray:
        H = len(self.horizons)
        if len(fills) == 0:
            return np.empty((0, H))
        ts, mid = self._book_path()
        i0 = np.searchsorted(ts, fills["ts"], side="right") - 1
        m0 = mid[np.clip(i0, 0, len(ts) - 1)]
        out = np.empty((len(fills), H))
        for h_i, h in enumerate(self.horizons):
            j = np.searchsorted(ts, fills["ts"] + h, side="right") - 1
            out[:, h_i] = fills["side"] * (mid[np.clip(j, 0, len(ts) - 1)] - m0)
        return out
