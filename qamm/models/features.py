"""
Market-state features computed once, vectorised, for every book update.

All features are *causal* (use only information up to and including row i).
They serve the fill-probability model, the markout model, the Bayesian and RL
quoting policies and the dashboard, so that every component sees the same
state description.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import MarketData

FEATURE_NAMES = [
    "spread_ticks",      # best ask − best bid in ticks
    "imb1",              # L1 imbalance (bid−ask)/(bid+ask)
    "imb3",              # L1..3 imbalance (falls back to imb1 for depth-1 feeds)
    "log_bid_sz1",       # log top-of-book bid size
    "log_ask_sz1",
    "trade_cnt_1s",      # trades in trailing 1 s
    "trade_cnt_5s",
    "signed_flow_1s",    # (buy vol − sell vol)/(total vol) trailing 1 s
    "signed_flow_5s",
    "vol_10s_ticks",     # realised vol: sqrt(Σ Δmid²) over trailing 10 s, in ticks
    "ret_1s_ticks",      # mid(t) − mid(t−1 s) in ticks
    "t_since_trade",     # seconds since last trade (capped at 10)
    "trade_rate_30s",    # trades per second trailing 30 s (Hawkes-intensity proxy)
]

ORDER_FEATURE_NAMES = [
    "side",              # +1 bid, −1 ask
    "dist_ticks",        # distance of our price from the best on our side (0 = at best)
    "queue_ahead",       # volume ahead of us (base units)
    "queue_ratio",       # queue_ahead / top-of-book size on our side
    "log_queue",         # log1p(queue_ahead / lot)
    "same_side_imb",     # imbalance signed in our favour (positive = more volume on our side)
]


def compute_features(md: MarketData) -> np.ndarray:
    """Return (N, len(FEATURE_NAMES)) float32 array aligned with md.book rows."""
    b, tr = md.book, md.trades
    ts = b.ts
    tick = md.tick_size
    N = b.n
    mid = b.mid
    depth = b.depth
    out = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float32)

    out[:, 0] = b.spread / tick
    out[:, 1] = b.imbalance(1)
    out[:, 2] = b.imbalance(min(3, depth))
    out[:, 3] = np.log(np.maximum(b.bid_sz[:, 0], 1e-9))
    out[:, 4] = np.log(np.maximum(b.ask_sz[:, 0], 1e-9))

    # trade windows via searchsorted (trades sorted by ts)
    tts = tr.ts
    ptr_now = np.searchsorted(tts, ts, side="right")
    def window(w):
        return np.searchsorted(tts, ts - w, side="right")
    p1, p5, p30 = window(1.0), window(5.0), window(30.0)
    out[:, 5] = ptr_now - p1
    out[:, 6] = ptr_now - p5
    if tr.n:
        buy_cum = np.concatenate([[0.0], np.cumsum(np.where(tr.side > 0, tr.size, 0.0))])
        sell_cum = np.concatenate([[0.0], np.cumsum(np.where(tr.side < 0, tr.size, 0.0))])
        def flow(p):
            bv = buy_cum[ptr_now] - buy_cum[p]
            sv = sell_cum[ptr_now] - sell_cum[p]
            return (bv - sv) / np.maximum(bv + sv, 1e-12)
        out[:, 7] = flow(p1)
        out[:, 8] = flow(p5)
        last_trade_ts = np.where(ptr_now > 0, tts[np.maximum(ptr_now - 1, 0)], -np.inf)
        out[:, 11] = np.minimum(ts - last_trade_ts, 10.0)
    else:
        out[:, 11] = 10.0
    out[:, 12] = (ptr_now - p30) / 30.0

    # realised vol & return from mid path
    dmid2 = np.concatenate([[0.0], np.diff(mid) ** 2])
    cum = np.cumsum(dmid2)
    i10 = np.searchsorted(ts, ts - 10.0, side="left")
    out[:, 9] = np.sqrt(np.maximum(cum - cum[i10], 0.0)) / tick
    i1 = np.searchsorted(ts, ts - 1.0, side="left")
    out[:, 10] = (mid - mid[i1]) / tick
    return out


class OnlineFeatures:
    """Incremental version of `compute_features` for closed-loop simulation.

    Feed every book update with `update(...)`; `vector()` returns the same 13
    features as the vectorised function (same definitions, same windows)."""

    def __init__(self, tick: float):
        from collections import deque
        self.tick = tick
        self.trades = deque()      # (ts, side, size)
        self.mids = deque()        # (ts, mid)
        self.dmid2 = deque()       # (ts, Δmid²)
        self.last_mid = None
        self.last_trade_ts = -np.inf
        self.cur = None

    def update(self, ts, bid_px1, bid_sz1, ask_px1, ask_sz1, imb3, new_trades):
        for (tts, side, size) in new_trades:
            self.trades.append((tts, side, size))
            self.last_trade_ts = tts
        mid = 0.5 * (bid_px1 + ask_px1)
        if self.last_mid is not None:
            self.dmid2.append((ts, (mid - self.last_mid) ** 2))
        self.last_mid = mid
        self.mids.append((ts, mid))
        while self.trades and self.trades[0][0] <= ts - 30.0:
            self.trades.popleft()
        while self.mids and self.mids[0][0] < ts - 10.5:
            self.mids.popleft()
        while self.dmid2 and self.dmid2[0][0] <= ts - 10.0:
            self.dmid2.popleft()
        self.cur = (ts, bid_px1, bid_sz1, ask_px1, ask_sz1, imb3, mid)

    def vector(self) -> np.ndarray:
        ts, bp, bs, ap, asz, imb3, mid = self.cur
        v = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        v[0] = (ap - bp) / self.tick
        v[1] = (bs - asz) / max(bs + asz, 1e-12)
        v[2] = imb3
        v[3] = np.log(max(bs, 1e-9)); v[4] = np.log(max(asz, 1e-9))
        n1 = n5 = 0; b1 = s1 = b5 = s5 = 0.0
        for (tts, side, size) in self.trades:
            if tts > ts - 5.0:
                n5 += 1
                if side > 0: b5 += size
                else: s5 += size
                if tts > ts - 1.0:
                    n1 += 1
                    if side > 0: b1 += size
                    else: s1 += size
        v[5], v[6] = n1, n5
        v[7] = (b1 - s1) / max(b1 + s1, 1e-12)
        v[8] = (b5 - s5) / max(b5 + s5, 1e-12)
        v[9] = np.sqrt(sum(d for _, d in self.dmid2)) / self.tick
        # mid 1 s ago: last mid with ts <= ts - 1
        m1 = mid
        for (mts, m) in reversed(self.mids):
            if mts <= ts - 1.0:
                m1 = m
                break
        else:
            m1 = self.mids[0][1] if self.mids else mid
        v[10] = (mid - m1) / self.tick
        v[11] = min(ts - self.last_trade_ts, 10.0)
        v[12] = len(self.trades) / 30.0
        return v


def order_features(side: int, dist_ticks: float, queue_ahead: float, top_sz_same: float,
                   imb1: float, lot: float = 0.001) -> np.ndarray:
    """Per-order features for a (hypothetical) passive order."""
    return np.array([
        side, dist_ticks, queue_ahead,
        queue_ahead / max(top_sz_same, 1e-9),
        np.log1p(queue_ahead / lot),
        side * imb1,
    ], dtype=np.float32)
