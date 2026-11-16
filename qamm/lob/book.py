"""
Aggregate limit order book with *queue-position tracking* for the market maker's
own orders.

The public order flow is modelled at the level of aggregate size per price
level (as in any L2 feed). The market maker's orders are tracked individually:
each knows how much volume sits *ahead* of it in the FIFO queue at its price.

Queue mechanics (price-time priority):
  * limit add at our price           -> joins behind us (queue_ahead unchanged)
  * cancel at our price of volume c   -> a share c·q/(L−s) is assumed to come
                                         from ahead of us (uniform cancellation)
  * market order consuming our level  -> eats queue_ahead first, then us, then
                                         volume behind us
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class Side(IntEnum):
    BID = 1
    ASK = -1


@dataclass
class TrackedOrder:
    oid: int
    side: Side
    price: float
    size: float
    queue_ahead: float
    placed_ts: float
    remaining: float = field(init=False)
    filled: float = 0.0
    fills: List[Tuple[float, float]] = field(default_factory=list)  # (ts, qty)

    def __post_init__(self):
        self.remaining = self.size

    @property
    def is_done(self) -> bool:
        return self.remaining <= 1e-12


class OrderBook:
    """L2 book with two sorted sides. Bids descending, asks ascending.

    Each side is a list of [price, size]. `size` includes the market maker's
    resting volume at that price (so snapshots look like a real feed)."""

    EPS = 1e-9

    def __init__(self, tick_size: float, n_levels: int = 10, max_internal_levels: Optional[int] = None):
        self.tick = float(tick_size)
        self.n_levels = int(n_levels)
        self.max_levels = max_internal_levels or (n_levels + 8)
        self.bids: List[List[float]] = []
        self.asks: List[List[float]] = []
        self.orders: Dict[int, TrackedOrder] = {}
        self._next_oid = 1
        self.last_trade_price: Optional[float] = None

    # ------------------------------------------------------------------ utils
    def _side(self, side: Side) -> List[List[float]]:
        return self.bids if side == Side.BID else self.asks

    def round_px(self, p: float) -> float:
        return round(round(p / self.tick) * self.tick, 10)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else float("nan")

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else float("nan")

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_ticks(self) -> int:
        return int(round(self.spread / self.tick))

    def level_size(self, side: Side, price: float) -> float:
        for p, s in self._side(side):
            if abs(p - price) < self.EPS:
                return s
        return 0.0

    def mm_size_at(self, side: Side, price: float) -> float:
        return sum(o.remaining for o in self.orders.values()
                   if o.side == side and abs(o.price - price) < self.EPS)

    def _find_level(self, side: Side, price: float) -> int:
        lv = self._side(side)
        for i, (p, _) in enumerate(lv):
            if abs(p - price) < self.EPS:
                return i
        return -1

    def _insert_level(self, side: Side, price: float, size: float) -> int:
        lv = self._side(side)
        # keep sorted: bids desc, asks asc
        i = 0
        if side == Side.BID:
            while i < len(lv) and lv[i][0] > price:
                i += 1
        else:
            while i < len(lv) and lv[i][0] < price:
                i += 1
        lv.insert(i, [price, size])
        if len(lv) > self.max_levels:
            # drop the deepest level unless an MM order rests there
            deep = lv[-1][0]
            if self.mm_size_at(side, deep) <= 0:
                lv.pop()
        return i

    def imbalance(self, levels: int = 1) -> float:
        b = sum(s for _, s in self.bids[:levels])
        a = sum(s for _, s in self.asks[:levels])
        return (b - a) / max(b + a, 1e-12)

    def depth_arrays(self, k: Optional[int] = None):
        k = k or self.n_levels
        import numpy as np
        bp = np.full(k, np.nan); bs = np.zeros(k); ap = np.full(k, np.nan); asz = np.zeros(k)
        for i, (p, s) in enumerate(self.bids[:k]):
            bp[i], bs[i] = p, s
        for i, (p, s) in enumerate(self.asks[:k]):
            ap[i], asz[i] = p, s
        return bp, bs, ap, asz

    # ---------------------------------------------------------- public flow
    def add_limit(self, side: Side, price: float, size: float) -> None:
        """Public (anonymous) limit order add. Joins the back of the queue."""
        i = self._find_level(side, price)
        if i >= 0:
            self._side(side)[i][1] += size
        else:
            self._insert_level(side, price, size)

    def cancel_public(self, side: Side, price: float, size: float) -> float:
        """Public cancel of `size` at (side, price). Only non-MM volume can be
        cancelled. Returns actually cancelled volume. Updates queue_ahead of
        MM orders at the level under uniform-cancellation assumption."""
        i = self._find_level(side, price)
        if i < 0:
            return 0.0
        lv = self._side(side)
        L = lv[i][1]
        mm = self.mm_size_at(side, price)
        public = max(L - mm, 0.0)
        c = min(size, public)
        if c <= 0:
            return 0.0
        for o in self.orders.values():
            if o.side == side and abs(o.price - price) < self.EPS and o.queue_ahead > 0:
                frac = o.queue_ahead / public
                o.queue_ahead = max(0.0, o.queue_ahead - c * frac)
        lv[i][1] = L - c
        if lv[i][1] <= self.EPS and mm <= 0:
            lv.pop(i)
        return c

    def market_order(self, side: Side, size: float, ts: float) -> List[Tuple[float, float]]:
        """Aggressive order. side=BID means a *buy* market order that hits the asks.
        Returns list of (price, executed size) per level consumed. Fills MM orders
        respecting queue position."""
        opp = Side.ASK if side == Side.BID else Side.BID
        lv = self._side(opp)
        remaining = size
        prints: List[Tuple[float, float]] = []
        while remaining > self.EPS and lv:
            price, L = lv[0][0], lv[0][1]
            take_total = min(remaining, L)
            # walk the FIFO queue at this level for MM orders
            mm_orders = sorted((o for o in self.orders.values()
                                if o.side == opp and abs(o.price - price) < self.EPS and not o.is_done),
                               key=lambda o: o.queue_ahead)
            left = take_total
            for o in mm_orders:
                if left <= self.EPS:
                    break
                ahead = min(o.queue_ahead, left)
                o.queue_ahead -= ahead
                left -= ahead
                if o.queue_ahead <= self.EPS and left > self.EPS:
                    f = min(o.remaining, left)
                    o.remaining -= f
                    o.filled += f
                    o.fills.append((ts, f))
                    left -= f
            lv[0][1] = L - take_total
            remaining -= take_total
            prints.append((price, take_total))
            self.last_trade_price = price
            if lv[0][1] <= self.EPS:
                lv.pop(0)
        # purge completed MM orders
        for oid in [k for k, o in self.orders.items() if o.is_done]:
            del self.orders[oid]
        return prints

    # ---------------------------------------------------------- MM orders
    def place(self, side: Side, price: float, size: float, ts: float) -> TrackedOrder:
        price = self.round_px(price)
        # must be passive
        if side == Side.BID and self.asks and price >= self.best_ask - self.EPS:
            raise ValueError("bid would cross the book")
        if side == Side.ASK and self.bids and price <= self.best_bid + self.EPS:
            raise ValueError("ask would cross the book")
        i = self._find_level(side, price)
        if i >= 0:
            ahead = self._side(side)[i][1]
            self._side(side)[i][1] += size
        else:
            ahead = 0.0
            self._insert_level(side, price, size)
        o = TrackedOrder(self._next_oid, side, price, size, ahead, ts)
        self._next_oid += 1
        self.orders[o.oid] = o
        return o

    def cancel(self, oid: int) -> Optional[TrackedOrder]:
        o = self.orders.pop(oid, None)
        if o is None:
            return None
        i = self._find_level(o.side, o.price)
        if i >= 0:
            lv = self._side(o.side)
            lv[i][1] = max(0.0, lv[i][1] - o.remaining)
            if lv[i][1] <= self.EPS:
                lv.pop(i)
        return o

    def cancel_all(self) -> List[TrackedOrder]:
        return [self.cancel(oid) for oid in list(self.orders)]
