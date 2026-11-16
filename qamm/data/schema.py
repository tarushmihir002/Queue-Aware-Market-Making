"""
Canonical in-memory market-data schema shared by the synthetic simulator, the
Binance loader and any user-supplied L2 feed.

Everything is stored as flat NumPy arrays (one row per book update / trade) so
that the backtester and the models never depend on the data source.

    L2Book   – book snapshots after every event
    Trades   – executed trades with aggressor side
    MarketData = L2Book + Trades + metadata
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class L2Book:
    ts: np.ndarray            # (N,) float64 seconds
    bid_px: np.ndarray        # (N, K) best bid first
    bid_sz: np.ndarray        # (N, K)
    ask_px: np.ndarray        # (N, K) best ask first
    ask_sz: np.ndarray        # (N, K)

    @property
    def n(self) -> int:
        return len(self.ts)

    @property
    def depth(self) -> int:
        return self.bid_px.shape[1]

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.bid_px[:, 0] + self.ask_px[:, 0])

    @property
    def spread(self) -> np.ndarray:
        return self.ask_px[:, 0] - self.bid_px[:, 0]

    def imbalance(self, levels: int = 1) -> np.ndarray:
        """(bid volume − ask volume) / (bid + ask) over the first `levels` levels. ∈ [-1, 1]."""
        b = self.bid_sz[:, :levels].sum(1)
        a = self.ask_sz[:, :levels].sum(1)
        return (b - a) / np.maximum(b + a, 1e-12)

    def microprice(self) -> np.ndarray:
        b, a = self.bid_sz[:, 0], self.ask_sz[:, 0]
        w = a / np.maximum(a + b, 1e-12)
        return w * self.bid_px[:, 0] + (1 - w) * self.ask_px[:, 0]

    def slice(self, i0: int, i1: int) -> "L2Book":
        return L2Book(self.ts[i0:i1], self.bid_px[i0:i1], self.bid_sz[i0:i1],
                      self.ask_px[i0:i1], self.ask_sz[i0:i1])


@dataclass
class Trades:
    ts: np.ndarray            # (M,) seconds
    price: np.ndarray         # (M,)
    size: np.ndarray          # (M,)
    side: np.ndarray          # (M,) int8: +1 buyer-initiated (hits ask), -1 seller-initiated (hits bid)

    @property
    def n(self) -> int:
        return len(self.ts)


@dataclass
class MarketData:
    book: L2Book
    trades: Trades
    symbol: str = "SYNTH"
    tick_size: float = 0.1
    source: str = "synthetic"
    meta: dict = field(default_factory=dict)

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        ev = self.meta.get("events")
        if ev is not None and len(ev):
            extra["events"] = np.asarray(ev)
        np.savez_compressed(
            path,
            ts=self.book.ts, bid_px=self.book.bid_px, bid_sz=self.book.bid_sz,
            ask_px=self.book.ask_px, ask_sz=self.book.ask_sz,
            t_ts=self.trades.ts, t_price=self.trades.price, t_size=self.trades.size,
            t_side=self.trades.side,
            symbol=np.array(self.symbol), tick_size=np.array(self.tick_size),
            source=np.array(self.source), **extra,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MarketData":
        z = np.load(Path(path), allow_pickle=False)
        book = L2Book(z["ts"], z["bid_px"], z["bid_sz"], z["ask_px"], z["ask_sz"])
        trades = Trades(z["t_ts"], z["t_price"], z["t_size"], z["t_side"])
        md = cls(book, trades, symbol=str(z["symbol"]), tick_size=float(z["tick_size"]),
                 source=str(z["source"]))
        if "events" in z.files:
            md.meta["events"] = z["events"]
        return md

    def summary(self) -> dict:
        b = self.book
        return {
            "symbol": self.symbol, "source": self.source, "tick_size": self.tick_size,
            "n_book_updates": b.n, "n_trades": self.trades.n,
            "duration_s": float(b.ts[-1] - b.ts[0]) if b.n else 0.0,
            "mean_spread_ticks": float(np.mean(b.spread) / self.tick_size) if b.n else np.nan,
            "mean_top_depth": float(np.mean(b.bid_sz[:, 0] + b.ask_sz[:, 0]) / 2) if b.n else np.nan,
            "mid_start": float(b.mid[0]) if b.n else np.nan,
            "mid_end": float(b.mid[-1]) if b.n else np.nan,
        }
