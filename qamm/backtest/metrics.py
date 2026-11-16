"""Performance metrics and the P&L decomposition for BacktestResult."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .engine import BacktestResult


def summarize_result(r: BacktestResult, tick: float, as_horizon: float = 1.0, quote_size: float = 1.0) -> Dict[str, float]:
    f = r.fills
    n = len(f)
    pnl = r.mtm_pnl
    final = float(pnl[-1]) if len(pnl) else 0.0
    spread_rev = float(np.sum(f["side"] * f["qty"] * (f["mid_at_fill"] - f["price"]))) if n else 0.0
    fees = r.fees_paid
    inv_pnl = final - spread_rev + fees           # exact residual
    hi = int(np.argmin(np.abs(r.markout_horizons - as_horizon)))
    adv = float(-np.sum(f["qty"] * r.markouts[:, hi])) if n else 0.0
    dur = float(r.ts[-1] - r.ts[0]) if len(r.ts) > 1 else 1.0
    # per-minute PnL series for Sharpe
    if len(r.ts) > 2:
        mins = np.floor((r.ts - r.ts[0]) / 60.0)
        _, first = np.unique(mins, return_index=True)
        p_min = np.diff(pnl[np.r_[first, len(pnl) - 1]])
        sharpe = float(p_min.mean() / (p_min.std() + 1e-12) * np.sqrt(60 * 24 * 365)) if len(p_min) > 2 else float("nan")
    else:
        sharpe = float("nan")
    peak = np.maximum.accumulate(pnl) if len(pnl) else np.array([0.0])
    mdd = float(np.max(peak - pnl)) if len(pnl) else 0.0
    n_quotes = r.n_bid_quotes + r.n_ask_quotes
    out = {
        "strategy": r.name,
        "net_pnl": final - r.liquidation_cost,
        "mtm_pnl": final,
        "spread_revenue": spread_rev,
        "inventory_pnl": inv_pnl,
        "fees": fees,
        f"adverse_selection_{as_horizon:g}s": adv,
        "inventory_cost": r.inventory_cost,
        "liquidation_cost": r.liquidation_cost,
        "objective": spread_rev - adv - fees - r.inventory_cost,
        "n_fills": n,
        "n_quotes": n_quotes,
        "fill_rate_per_quote": n / max(n_quotes, 1),
        "fills_per_min": n / max(dur / 60.0, 1e-9),
        "traded_volume": float(f["qty"].sum()) if n else 0.0,
        "avg_markout_ticks": float(np.mean(r.markouts[:, hi]) / tick) if n else float("nan"),
        "avg_spread_capture_ticks": float(np.mean(f["side"] * (f["mid_at_fill"] - f["price"])) / tick) if n else float("nan"),
        "avg_queue_at_place": float(np.mean(f["queue_at_place"])) if n else float("nan"),
        "avg_time_to_fill_s": float(np.mean(f["ts"] - f["placed_ts"])) if n else float("nan"),
        "mean_abs_inventory": float(np.mean(np.abs(r.inventory))) if len(r.inventory) else 0.0,
        "max_abs_inventory": float(np.max(np.abs(r.inventory))) if len(r.inventory) else 0.0,
        "sharpe_annualised": sharpe,
        "max_drawdown": mdd,
        "duration_s": dur,
        "pnl_per_hour": (final - r.liquidation_cost) / max(dur / 3600.0, 1e-9),
        "wall_time_s": r.wall_time_s,
    }
    return out


def compare_results(results: List[BacktestResult], tick: float, **kw) -> "pd.DataFrame":
    import pandas as pd
    rows = [summarize_result(r, tick, **kw) for r in results]
    return pd.DataFrame(rows).set_index("strategy")
