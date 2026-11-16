"""
Binance public market-data archive loader (https://data.binance.vision).

Binance publishes, per symbol and per day, zipped CSVs with no authentication:

  spot/daily/aggTrades/<SYM>/<SYM>-aggTrades-YYYY-MM-DD.zip      every trade
  spot/daily/bookTicker/<SYM>/<SYM>-bookTicker-YYYY-MM-DD.zip    every best-bid/ask update
  futures/um/daily/bookDepth/<SYM>/...                           (USD-M futures only) depth at ±pct levels

Full multi-level L2 history is *not* in the public archive (it requires a
websocket recorder or a vendor such as Tardis / the crypto-lob-stream dataset —
see `qamm.data.loader.load_l2_csv` for those). The bookTicker + aggTrades
combination gives exact top-of-book queue size and every trade, which is what
the queue-position fill model and the markout estimator need. Deeper levels are
filled with NaN/0 and the depth-aware features fall back gracefully.

Usage
-----
    from qamm.data.binance import download_binance_day, load_binance_day
    download_binance_day("BTCUSDT", "2024-05-01", out_dir="data/binance")
    md = load_binance_day("BTCUSDT", "2024-05-01", data_dir="data/binance")
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from .schema import L2Book, MarketData, Trades

BINANCE_VISION = "https://data.binance.vision/data"

# Binance spot tick sizes for the symbols in the brief (as of 2024/2025)
TICK_SIZES = {"BTCUSDT": 0.01, "ETHUSDT": 0.01}

AGGTRADES_COLS = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker", "best_match"]
BOOKTICKER_COLS = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty",
                   "transaction_time", "event_time"]


def _url(kind: str, symbol: str, day: str, market: str = "spot") -> str:
    base = f"{BINANCE_VISION}/{market}/daily/{kind}/{symbol}"
    return f"{base}/{symbol}-{kind}-{day}.zip"


def download_binance_day(symbol: str, day: str, out_dir: str | Path = "data/binance",
                         kinds=("aggTrades", "bookTicker"), market: str = "spot",
                         timeout: int = 120, force: bool = False) -> dict:
    """Download the daily archives. Returns {kind: local_path}. Files are ~50–300 MB zipped."""
    import requests

    out = Path(out_dir) / symbol
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind in kinds:
        dest = out / f"{symbol}-{kind}-{day}.zip"
        if dest.exists() and not force:
            paths[kind] = dest
            continue
        url = _url(kind, symbol, day, market)
        print(f"downloading {url}")
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {url} — check symbol/date; bookTicker daily "
                               f"archives exist for spot from 2023-01 onwards.")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        paths[kind] = dest
    return paths


def _read_zip_csv(path: Path, columns: list[str]) -> pl.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    # newer archives include a header row, older ones don't
    head = raw[:200].decode("utf-8", errors="ignore")
    has_header = not head[0].isdigit()
    df = pl.read_csv(io.BytesIO(raw), has_header=has_header, new_columns=None if has_header else columns)
    if has_header:
        df = df.rename({c: n for c, n in zip(df.columns, columns)})
    return df


def load_binance_day(symbol: str, day: str, data_dir: str | Path = "data/binance",
                     max_rows: Optional[int] = None, start_s: float = 0.0,
                     duration_s: Optional[float] = None) -> MarketData:
    """Build MarketData (depth-1 book + trades) from the daily bookTicker/aggTrades archives.

    `start_s`/`duration_s` select a window inside the day (seconds since midnight UTC)."""
    d = Path(data_dir) / symbol
    bt = _read_zip_csv(d / f"{symbol}-bookTicker-{day}.zip", BOOKTICKER_COLS)
    at = _read_zip_csv(d / f"{symbol}-aggTrades-{day}.zip", AGGTRADES_COLS)

    # timestamps: ms (older) or µs (2025+) epoch -> seconds
    def to_s(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        return x / 1e6 if np.nanmedian(x) > 1e14 else x / 1e3

    bts = to_s(bt["transaction_time"].to_numpy())
    ats = to_s(at["ts"].to_numpy())
    day0 = np.floor(bts[0] / 86400) * 86400
    bts -= day0
    ats -= day0
    t1 = start_s + (duration_s if duration_s else 1e12)
    bm = (bts >= start_s) & (bts < t1)
    am = (ats >= start_s) & (ats < t1)
    bt, bts = bt.filter(pl.Series(bm)), bts[bm]
    at, ats = at.filter(pl.Series(am)), ats[am]
    if max_rows:
        bt, bts = bt.head(max_rows), bts[:max_rows]
        am2 = ats <= bts[-1]
        at, ats = at.filter(pl.Series(am2)), ats[am2]

    K = 1
    bid_px = bt["best_bid_price"].to_numpy().astype(float).reshape(-1, K)
    bid_sz = bt["best_bid_qty"].to_numpy().astype(float).reshape(-1, K)
    ask_px = bt["best_ask_price"].to_numpy().astype(float).reshape(-1, K)
    ask_sz = bt["best_ask_qty"].to_numpy().astype(float).reshape(-1, K)
    # drop crossed / stale rows
    ok = (ask_px[:, 0] > bid_px[:, 0]) & (bid_sz[:, 0] > 0) & (ask_sz[:, 0] > 0)
    book = L2Book(bts[ok] - start_s, bid_px[ok], bid_sz[ok], ask_px[ok], ask_sz[ok])

    # aggTrades: is_buyer_maker == True  -> buyer was passive -> seller-initiated (side = -1)
    ibm = at["is_buyer_maker"].to_numpy()
    if ibm.dtype.kind in "OU":
        ibm = np.char.lower(ibm.astype(str)) == "true"
    side = np.where(ibm.astype(bool), -1, 1).astype(np.int8)
    trades = Trades(ats - start_s, at["price"].to_numpy().astype(float), at["qty"].to_numpy().astype(float), side)
    return MarketData(book, trades, symbol=symbol, tick_size=TICK_SIZES.get(symbol, 0.01), source="binance",
                      meta={"day": day, "start_s": start_s})
