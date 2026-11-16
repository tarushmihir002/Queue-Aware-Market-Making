#!/usr/bin/env python
"""
Download public Binance L2 (bookTicker) + trade (aggTrades) daily archives and
cache them as a MarketData .npz.

No API key needed — the data comes from https://data.binance.vision.

Examples
--------
  python scripts/download_binance.py BTCUSDT 2024-05-01
  python scripts/download_binance.py ETHUSDT 2024-05-01 --window 0 3600 --save results/eth_1h.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qamm.data import download_binance_day, load_binance_day


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol")
    ap.add_argument("day", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "binance"))
    ap.add_argument("--window", nargs=2, type=float, metavar=("START_S", "DURATION_S"), default=None,
                    help="seconds since 00:00 UTC; default = whole day (large!)")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--save", default=None, help="also save the built MarketData as .npz")
    args = ap.parse_args()

    paths = download_binance_day(args.symbol, args.day, out_dir=args.data_dir)
    print("downloaded:", {k: str(v) for k, v in paths.items()})
    start, dur = (args.window if args.window else (0.0, None))
    md = load_binance_day(args.symbol, args.day, data_dir=args.data_dir, start_s=start,
                          duration_s=dur, max_rows=args.max_rows)
    print(md.summary())
    if args.save:
        p = md.save(args.save)
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()
