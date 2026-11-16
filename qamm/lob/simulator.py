"""
Event-driven limit-order-book simulator driven by a 6-dimensional Hawkes process.

Event types:
    0 MB  market buy   (hits asks)      1 MS  market sell  (hits bids)
    2 LB  limit buy add                 3 LS  limit sell add
    4 CB  cancel on bid side            5 CS  cancel on ask side

The Hawkes process generates *when* and *what kind* of event happens. The
simulator then draws the *size* and *price level* and applies it to the book.
Because the kernel couples market sells → bid cancels → further market sells,
passive bids get filled preferentially right before the price drops: that is
the adverse-selection mechanism this project studies.

Two ways to use it:
  * `run()`           : generate a full MarketData record (book snapshots + trades)
  * `step()` loop     : advance one event at a time while a strategy places and
                        cancels orders in the live book (closed-loop backtest)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

from ..data.schema import L2Book, MarketData, Trades
from ..hawkes import HawkesParams, MultivariateHawkes
from .book import OrderBook, Side


class EventType(IntEnum):
    MB = 0
    MS = 1
    LB = 2
    LS = 3
    CB = 4
    CS = 5


EVENT_NAMES = ["MB", "MS", "LB", "LS", "CB", "CS"]


@dataclass
class Event:
    ts: float
    kind: int
    side: Side
    price: float
    size: float
    prints: List[Tuple[float, float]]  # trade prints for market orders


class HawkesLOBSimulator:
    def __init__(self, cfg: dict, seed: int = 0, hawkes_params: Optional[HawkesParams] = None):
        self.cfg = cfg
        inst, syn = cfg["instrument"], cfg["synthetic"]
        self.tick = float(inst["tick_size"])
        self.n_levels = int(inst["n_levels"])
        self.syn = syn
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.hp = hawkes_params or HawkesParams.from_config(cfg["hawkes"])
        self.hawkes = MultivariateHawkes(self.hp)
        self.book = OrderBook(self.tick, self.n_levels)
        self.T = float(syn["duration_s"])
        self._init_book(float(inst["initial_mid"]))
        self.times: np.ndarray = np.empty(0)
        self.kinds: np.ndarray = np.empty(0, dtype=np.int64)
        self.ptr = 0
        self.t = 0.0
        # recording buffers
        self.rec_ts: List[float] = []
        self.rec_bp: List[np.ndarray] = []
        self.rec_bs: List[np.ndarray] = []
        self.rec_ap: List[np.ndarray] = []
        self.rec_as: List[np.ndarray] = []
        self.rec_trades: List[Tuple[float, float, float, int]] = []
        self.rec_events: List[Tuple[float, int, float, float]] = []

    # ------------------------------------------------------------------
    def _init_book(self, mid: float) -> None:
        b = self.book
        half = self.tick  # 2-tick initial spread
        bb = b.round_px(mid - half / 2)
        ba = b.round_px(bb + 2 * self.tick) if b.round_px(mid + half / 2) <= bb else b.round_px(mid + half / 2)
        for k in range(self.n_levels + 3):
            b.add_limit(Side.BID, b.round_px(bb - k * self.tick), self._level_seed_size())
            b.add_limit(Side.ASK, b.round_px(ba + k * self.tick), self._level_seed_size())

    def _level_seed_size(self) -> float:
        m = self.syn["initial_level_size"]
        return float(self.rng.lognormal(np.log(m), 0.5))

    def _replenish(self) -> None:
        """Keep at least n_levels on each side by appending deep levels."""
        b = self.book
        for side, lv, sgn in ((Side.BID, b.bids, -1), (Side.ASK, b.asks, +1)):
            while len(lv) < self.n_levels + 2:
                if lv:
                    p = b.round_px(lv[-1][0] + sgn * self.tick)
                else:  # side wiped out; rebuild around the other side
                    ref = b.best_ask if side == Side.BID else b.best_bid
                    p = b.round_px(ref + sgn * self.tick)
                b.add_limit(side, p, float(self.rng.lognormal(np.log(self.syn["depth_replenish_size"]), 0.5)))

    # ------------------------------------------------------------------
    def generate_events(self, T: Optional[float] = None) -> None:
        T = T or self.T
        self.times, self.kinds = self.hawkes.simulate(T, seed=self.seed)
        self.ptr = 0

    def _draw_market_size(self) -> float:
        s = self.rng.lognormal(np.log(self.syn["market_size_mean"]), self.syn["market_size_sigma"])
        return float(s)

    def _draw_limit_size(self) -> float:
        return float(self.rng.lognormal(np.log(self.syn["limit_size_mean"]), self.syn["limit_size_sigma"]))

    def _draw_limit_price(self, side: Side) -> float:
        b = self.book
        best = b.best_bid if side == Side.BID else b.best_ask
        sgn = 1 if side == Side.BID else -1
        if b.spread_ticks > 1 and self.rng.random() < self.syn["limit_improve_prob"]:
            # improve: place inside the spread, 1..spread-1 ticks better than best
            k = int(self.rng.integers(1, b.spread_ticks))
            return b.round_px(best + sgn * k * self.tick)
        k = int(self.rng.geometric(self.syn["limit_level_geom_p"]) - 1)
        k = min(k, self.n_levels - 1)
        return b.round_px(best - sgn * k * self.tick)

    def _apply(self, ts: float, kind: int) -> Event:
        b = self.book
        prints: List[Tuple[float, float]] = []
        if kind == EventType.MB or kind == EventType.MS:
            side = Side.BID if kind == EventType.MB else Side.ASK  # BID = buy aggressor
            size = self._draw_market_size()
            opp = b.asks if kind == EventType.MB else b.bids
            cap = sum(s for _, s in opp[:3])  # don't sweep more than 3 levels
            size = min(size, max(cap, 1e-6))
            prints = b.market_order(side, size, ts)
            px = prints[0][0] if prints else float("nan")
            agg = 1 if kind == EventType.MB else -1
            for p, q in prints:
                self.rec_trades.append((ts, p, q, agg))
            ev = Event(ts, kind, side, px, size, prints)
        elif kind == EventType.LB or kind == EventType.LS:
            side = Side.BID if kind == EventType.LB else Side.ASK
            size = self._draw_limit_size()
            px = self._draw_limit_price(side)
            b.add_limit(side, px, size)
            ev = Event(ts, kind, side, px, size, prints)
        else:
            side = Side.BID if kind == EventType.CB else Side.ASK
            lv = b._side(side)
            # choose a level ∝ public size (non-MM)
            pub = np.array([max(s - b.mm_size_at(side, p), 0.0) for p, s in lv[: self.n_levels]])
            tot = pub.sum()
            if tot <= 0:
                ev = Event(ts, kind, side, float("nan"), 0.0, prints)
            else:
                i = int(self.rng.choice(len(pub), p=pub / tot))
                px = lv[i][0]
                frac = float(np.clip(self.rng.beta(2, 2) * 2 * self.syn["cancel_frac_mean"], 0.02, 1.0))
                size = b.cancel_public(side, px, frac * pub[i])
                ev = Event(ts, kind, side, px, size, prints)
        self._replenish()
        return ev

    # ------------------------------------------------------------------
    def step(self) -> Optional[Event]:
        """Apply the next Hawkes event. Returns None when exhausted."""
        if self.ptr >= len(self.times):
            return None
        ts, kind = float(self.times[self.ptr]), int(self.kinds[self.ptr])
        self.ptr += 1
        self.t = ts
        ev = self._apply(ts, kind)
        self._record(ts, kind, ev)
        return ev

    def peek_next_time(self) -> float:
        return float(self.times[self.ptr]) if self.ptr < len(self.times) else float("inf")

    def _record(self, ts: float, kind: int, ev: Event) -> None:
        bp, bs, ap, asz = self.book.depth_arrays(self.n_levels)
        self.rec_ts.append(ts)
        self.rec_bp.append(bp); self.rec_bs.append(bs); self.rec_ap.append(ap); self.rec_as.append(asz)
        self.rec_events.append((ts, kind, ev.price, ev.size))

    def run(self, T: Optional[float] = None, progress: bool = False) -> MarketData:
        self.generate_events(T)
        n = len(self.times)
        for i in range(n):
            self.step()
            if progress and i % 50_000 == 0:
                print(f"  sim {i}/{n}")
        return self.to_market_data()

    def to_market_data(self) -> MarketData:
        book = L2Book(np.asarray(self.rec_ts), np.vstack(self.rec_bp), np.vstack(self.rec_bs),
                      np.vstack(self.rec_ap), np.vstack(self.rec_as))
        if self.rec_trades:
            tr = np.array(self.rec_trades, dtype=float)
            trades = Trades(tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3].astype(np.int8))
        else:
            trades = Trades(np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=np.int8))
        ev = np.array(self.rec_events, dtype=float) if self.rec_events else np.empty((0, 4))
        md = MarketData(book, trades, symbol=self.cfg["instrument"]["symbol"], tick_size=self.tick,
                        source="synthetic", meta={"hawkes": self.hp.to_dict(), "seed": self.seed})
        md.meta["events"] = ev
        return md
