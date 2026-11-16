"""
Execution models: how a market maker's passive orders get filled.

ReplayExecution
    Replays *recorded* market data (synthetic or real). Our orders are virtual:
    they don't change the book. Each order tracks the volume ahead of it in the
    FIFO queue, which is decremented by
        * trades at our price on the aggressing side (eat the queue, then us),
        * trades through our price (level swept -> we are filled),
        * cancellations at our price (a share q/L assumed to be ahead of us).
    This is the standard queue-position replay model and works identically for
    multi-level synthetic data and Binance top-of-book + trades.

ClosedLoopExecution
    Drives the HawkesLOBSimulator live; our orders *are* in the book and get
    filled by the simulator's exact FIFO logic (ground truth for the synthetic
    world, including any market impact of our own resting size).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data.schema import MarketData

OUTCOME_NONE, OUTCOME_FILL, OUTCOME_REQUOTE = 0, 1, 2
EPS = 1e-9


@dataclass
class Fill:
    ts: float
    oid: int
    side: int        # +1 we bought (our bid was hit), −1 we sold
    price: float
    qty: float
    queue_at_place: float
    placed_ts: float


@dataclass
class VirtualOrder:
    oid: int
    side: int
    price: float
    size: float
    queue_ahead: float
    placed_ts: float
    queue_at_place: float = field(init=False)
    remaining: float = field(init=False)

    def __post_init__(self):
        self.remaining = self.size
        self.queue_at_place = self.queue_ahead


# ---------------------------------------------------------------------------
# core queue-replay step (shared by ReplayExecution and the probe generator)
# ---------------------------------------------------------------------------
def _level_size(px_row: np.ndarray, sz_row: np.ndarray, price: float) -> float:
    m = np.abs(px_row - price) < EPS
    return float(sz_row[m].sum()) if m.any() else 0.0


def replay_order_step(o: VirtualOrder, md: MarketData, i: int, j0: int, j1: int,
                      fills: List[Tuple[float, float]]) -> int:
    """Advance virtual order `o` across book row i-1 -> i with trades [j0, j1).

    Appends (ts, qty) fills. Returns OUTCOME_FILL if fully filled, OUTCOME_REQUOTE if the
    best price on our side moved past us (we are no longer at the touch), else OUTCOME_NONE.
    """
    b, tr = md.book, md.trades
    side = o.side
    px_rows = b.bid_px if side > 0 else b.ask_px
    sz_rows = b.bid_sz if side > 0 else b.ask_sz
    traded_at_px = 0.0
    for j in range(j0, j1):
        if tr.side[j] == side:          # same-side aggressor cannot hit us
            continue
        tp = tr.price[j]
        swept = (tp < o.price - EPS) if side > 0 else (tp > o.price + EPS)
        if swept:
            # trade printed beyond our price => our whole level was consumed first
            q = o.remaining
            o.queue_ahead = 0.0
            o.remaining = 0.0
            fills.append((tr.ts[j], q))
            return OUTCOME_FILL
        if abs(tp - o.price) < EPS:
            v = tr.size[j]
            traded_at_px += v
            ahead = min(o.queue_ahead, v)
            o.queue_ahead -= ahead
            v -= ahead
            if v > EPS and o.queue_ahead <= EPS:
                f = min(o.remaining, v)
                o.remaining -= f
                fills.append((tr.ts[j], f))
                if o.remaining <= EPS:
                    return OUTCOME_FILL
    # cancellations at our price: size drop not explained by trades
    prev = _level_size(px_rows[i - 1], sz_rows[i - 1], o.price)
    cur = _level_size(px_rows[i], sz_rows[i], o.price)
    base = prev - traded_at_px
    if base > EPS:
        cancel = max(base - cur, 0.0)
        if cancel > 0 and o.queue_ahead > 0:
            o.queue_ahead = max(0.0, o.queue_ahead - cancel * min(1.0, o.queue_ahead / base))
    # opposite side crossed our price (e.g. gap move) => filled
    opp_best = b.ask_px[i, 0] if side > 0 else b.bid_px[i, 0]
    if (side > 0 and opp_best <= o.price + EPS) or (side < 0 and opp_best >= o.price - EPS):
        q = o.remaining
        o.queue_ahead = 0.0
        o.remaining = 0.0
        fills.append((b.ts[i], q))
        return OUTCOME_FILL
    # best on our side moved past us -> not at the touch any more
    best = px_rows[i, 0]
    if (side > 0 and best > o.price + EPS) or (side < 0 and best < o.price - EPS):
        return OUTCOME_REQUOTE
    return OUTCOME_NONE


# ---------------------------------------------------------------------------
class ExecutionModel(ABC):
    tick: float

    @property
    @abstractmethod
    def ts(self) -> float: ...

    @property
    @abstractmethod
    def best_bid(self) -> float: ...

    @property
    @abstractmethod
    def best_ask(self) -> float: ...

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @abstractmethod
    def level_size(self, side: int, price: float) -> float: ...

    @abstractmethod
    def top_sizes(self) -> Tuple[float, float]: ...

    @abstractmethod
    def place(self, side: int, price: float, size: float) -> VirtualOrder: ...

    @abstractmethod
    def cancel(self, oid: int) -> None: ...

    @abstractmethod
    def open_orders(self) -> Dict[int, VirtualOrder]: ...

    @abstractmethod
    def advance(self, dt: float) -> Tuple[List[Fill], bool]:
        """Advance market by dt seconds. Returns (fills, ended)."""

    @abstractmethod
    def feature_row(self) -> int:
        """Index into precomputed feature arrays for the current state."""

    @property
    @abstractmethod
    def end_ts(self) -> float: ...


class ReplayExecution(ExecutionModel):
    def __init__(self, md: MarketData, start_ts: float = 0.0):
        self.md = md
        self.tick = md.tick_size
        b = md.book
        self.i = int(np.searchsorted(b.ts, start_ts, side="left"))
        self.i = min(max(self.i, 1), b.n - 1)
        self._t = float(b.ts[self.i])
        self.trade_ptr = np.searchsorted(md.trades.ts, b.ts, side="right")
        self.orders: Dict[int, VirtualOrder] = {}
        self._oid = 1

    @property
    def ts(self) -> float:
        return self._t

    @property
    def end_ts(self) -> float:
        return float(self.md.book.ts[-1])

    @property
    def best_bid(self) -> float:
        return float(self.md.book.bid_px[self.i, 0])

    @property
    def best_ask(self) -> float:
        return float(self.md.book.ask_px[self.i, 0])

    def level_size(self, side: int, price: float) -> float:
        b = self.md.book
        return _level_size(b.bid_px[self.i] if side > 0 else b.ask_px[self.i],
                           b.bid_sz[self.i] if side > 0 else b.ask_sz[self.i], price)

    def top_sizes(self):
        b = self.md.book
        return float(b.bid_sz[self.i, 0]), float(b.ask_sz[self.i, 0])

    def feature_row(self) -> int:
        return self.i

    def place(self, side: int, price: float, size: float) -> VirtualOrder:
        q = self.level_size(side, price)  # join the back of the queue
        o = VirtualOrder(self._oid, side, price, size, q, self._t)
        self._oid += 1
        self.orders[o.oid] = o
        return o

    def cancel(self, oid: int) -> None:
        self.orders.pop(oid, None)

    def open_orders(self):
        return self.orders

    def advance(self, dt: float) -> Tuple[List[Fill], bool]:
        b = self.md.book
        target = self._t + dt
        fills: List[Fill] = []
        i_start = self.i
        while self.i + 1 < b.n and b.ts[self.i + 1] <= target:
            self.i += 1
            j0, j1 = self.trade_ptr[self.i - 1], self.trade_ptr[self.i]
            done = []
            for oid, o in self.orders.items():
                fl: List[Tuple[float, float]] = []
                out = replay_order_step(o, self.md, self.i, j0, j1, fl)
                for ts_, q in fl:
                    fills.append(Fill(ts_, oid, o.side, o.price, q, o.queue_at_place, o.placed_ts))
                if out == OUTCOME_FILL:
                    done.append(oid)
            for oid in done:
                del self.orders[oid]
        tr = self.md.trades
        j0, j1 = self.trade_ptr[i_start], self.trade_ptr[self.i]
        self.last_trades = [(float(tr.ts[j]), float(tr.price[j]), float(tr.size[j]), int(tr.side[j]))
                            for j in range(j0, j1)]
        self._t = min(target, float(b.ts[-1]))
        ended = self.i + 1 >= b.n
        return fills, ended


class ClosedLoopExecution(ExecutionModel):
    def __init__(self, sim):
        from ..lob.book import Side
        self.Side = Side
        self.sim = sim
        self.tick = sim.tick
        if len(sim.times) == 0:
            sim.generate_events()
        self._t = 0.0
        self._map: Dict[int, VirtualOrder] = {}  # oid -> VirtualOrder mirror
        self._fill_ptr: Dict[int, int] = {}
        self._trade_ptr = len(sim.rec_trades)
        self._row_ptr = len(sim.rec_ts)
        self.last_trades: list = []

    @property
    def ts(self) -> float:
        return self._t

    @property
    def end_ts(self) -> float:
        return float(self.sim.times[-1]) if len(self.sim.times) else 0.0

    @property
    def best_bid(self) -> float:
        return self.sim.book.best_bid

    @property
    def best_ask(self) -> float:
        return self.sim.book.best_ask

    def level_size(self, side: int, price: float) -> float:
        return self.sim.book.level_size(self.Side.BID if side > 0 else self.Side.ASK, price)

    def top_sizes(self):
        b = self.sim.book
        return (b.bids[0][1] if b.bids else 0.0), (b.asks[0][1] if b.asks else 0.0)

    def feature_row(self) -> int:
        return max(len(self.sim.rec_ts) - 1, 0)

    def place(self, side: int, price: float, size: float) -> VirtualOrder:
        o = self.sim.book.place(self.Side.BID if side > 0 else self.Side.ASK, price, size, self._t)
        v = VirtualOrder(o.oid, side, o.price, size, o.queue_ahead, self._t)
        self._map[o.oid] = v
        self._fill_ptr[o.oid] = 0
        return v

    def cancel(self, oid: int) -> None:
        self.sim.book.cancel(oid)
        self._map.pop(oid, None)
        self._fill_ptr.pop(oid, None)

    def open_orders(self):
        # sync queue_ahead from the live book
        for oid, v in self._map.items():
            o = self.sim.book.orders.get(oid)
            if o is not None:
                v.queue_ahead = o.queue_ahead
                v.remaining = o.remaining
        return self._map

    def advance(self, dt: float) -> Tuple[List[Fill], bool]:
        target = self._t + dt
        fills: List[Fill] = []
        book = self.sim.book
        # snapshot the tracked orders so we can read fills of orders purged by the book
        while self.sim.peek_next_time() <= target:
            before = {oid: o for oid, o in book.orders.items()}
            self.sim.step()
            for oid, v in list(self._map.items()):
                o = book.orders.get(oid) or before.get(oid)
                if o is None:
                    continue
                k = self._fill_ptr[oid]
                for ts_, q in o.fills[k:]:
                    fills.append(Fill(ts_, oid, v.side, v.price, q, v.queue_at_place, v.placed_ts))
                self._fill_ptr[oid] = len(o.fills)
                v.remaining = o.remaining
                v.queue_ahead = o.queue_ahead
                if o.is_done:
                    del self._map[oid]
                    del self._fill_ptr[oid]
        rt = self.sim.rec_trades
        self.last_trades = [(t, p, q, s) for (t, p, q, s) in rt[self._trade_ptr:]]
        self._trade_ptr = len(rt)
        self._t = target
        return fills, self.sim.ptr >= len(self.sim.times)

    def new_rows(self):
        """Book rows recorded since the last call (for the online feature tracker)."""
        s = self.sim
        i0, i1 = self._row_ptr, len(s.rec_ts)
        self._row_ptr = i1
        return i0, i1
