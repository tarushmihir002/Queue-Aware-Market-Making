"""
Generic loaders that turn user-supplied files into the canonical MarketData.

Supported:
  * `.npz` written by MarketData.save
  * "wide" L2 CSV/parquet with columns
        ts, bid_px_1..K, bid_sz_1..K, ask_px_1..K, ask_sz_1..K
    and a trades CSV/parquet with  ts, price, size, side  (side ∈ {+1,-1} or
    'buy'/'sell'). This is the layout of most public LOB datasets
    (crypto-lob-stream, LOBSTER-style exports, Tardis book snapshots) after a
    trivial rename.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from .schema import L2Book, MarketData, Trades


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


def _to_seconds(ts: np.ndarray) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.float64)
    # heuristics: ns / us / ms epoch -> seconds
    m = np.nanmedian(ts)
    if m > 1e17:
        return ts / 1e9
    if m > 1e14:
        return ts / 1e6
    if m > 1e11:
        return ts / 1e3
    return ts


def load_l2_csv(book_path: str | Path, trades_path: str | Path, depth: Optional[int] = None,
                tick_size: float = 0.01, symbol: str = "UNKNOWN", source: str = "csv") -> MarketData:
    bdf = _read_table(book_path)
    tdf = _read_table(trades_path)
    cols = bdf.columns
    K = depth or max(int(c.split("_")[-1]) for c in cols if c.startswith("bid_px_"))
    ts = _to_seconds(bdf["ts"].to_numpy())
    bid_px = np.column_stack([bdf[f"bid_px_{i}"].to_numpy() for i in range(1, K + 1)]).astype(float)
    bid_sz = np.column_stack([bdf[f"bid_sz_{i}"].to_numpy() for i in range(1, K + 1)]).astype(float)
    ask_px = np.column_stack([bdf[f"ask_px_{i}"].to_numpy() for i in range(1, K + 1)]).astype(float)
    ask_sz = np.column_stack([bdf[f"ask_sz_{i}"].to_numpy() for i in range(1, K + 1)]).astype(float)
    t0 = ts[0]
    book = L2Book(ts - t0, bid_px, bid_sz, ask_px, ask_sz)
    tts = _to_seconds(tdf["ts"].to_numpy()) - t0
    side = tdf["side"].to_numpy()
    if side.dtype.kind in "OU":
        side = np.where(np.char.lower(side.astype(str)) == "buy", 1, -1)
    trades = Trades(tts, tdf["price"].to_numpy().astype(float), tdf["size"].to_numpy().astype(float),
                    side.astype(np.int8))
    return MarketData(book, trades, symbol=symbol, tick_size=tick_size, source=source)


def load_market_data(path: str | Path, **kw) -> MarketData:
    path = Path(path)
    if path.suffix == ".npz":
        return MarketData.load(path)
    raise ValueError("use load_l2_csv for csv/parquet (needs both book and trades files)")
