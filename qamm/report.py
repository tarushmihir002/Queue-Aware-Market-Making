"""
End-to-end orchestration: data -> models -> backtests -> a single results bundle.

`run_full_pipeline` is the one code path used by the CLI (`scripts/run_pipeline.py`),
the Streamlit dashboard and the test-suite, so every surface sees identical
numbers. `bundle_to_json` serialises a compact, plot-ready view for the HTML
dashboard and the paper.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .backtest import BacktestEngine, ReplayExecution, summarize_result
from .data import MarketData, generate_synthetic
from .hawkes import HawkesParams, fit_hawkes_mle
from .models import (BayesianQueueModel, CompetingRiskModel, MarkoutModel, aalen_johansen,
                     compute_features, evaluate_fill_model, generate_probe_dataset, markout_table)
from .strategy import (AvellanedaStoikov, BayesianQuoter, FixedSpreadStrategy, JoinTouchStrategy,
                       RLStrategy, calibrate_kappa)
from .utils import load_config, seed_everything


def _downsample(x: np.ndarray, n: int = 2000) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx]


def build_models(cfg: dict, md_train: MarketData, F_train: np.ndarray, seed: int = 0,
                 verbose: bool = False) -> dict:
    """Fit every model on the training market data."""
    fm = cfg["fill_model"]
    ds = generate_probe_dataset(md_train, n_probes=int(fm["n_probe_orders"]), max_wait_s=float(fm["max_wait_s"]),
                                order_size=float(cfg["strategy"]["quote_size"]),
                                horizons=tuple(cfg["strategy"]["markout_horizons_s"]), seed=seed,
                                features=F_train, lot=float(cfg["instrument"]["lot_size"]))
    cr = CompetingRiskModel(seed=seed).fit(ds.X, ds.time, ds.cause, verbose=verbose)
    mm = MarkoutModel(horizons=ds.horizons, seed=seed).fit(ds)
    qm = BayesianQueueModel(m0=float(cfg["synthetic"]["market_size_mean"]),
                            forget=0.995).fit_from_data(md_train)
    kappa = calibrate_kappa(ds, tau=float(cfg["strategy"].get("decision_horizon_s", 1.0)))
    return {"probes": ds, "cr": cr, "markout": mm, "queue": qm, "kappa": kappa}


def make_strategies(cfg: dict, models: dict, rl_agent=None) -> List:
    str50 = [
        JoinTouchStrategy(cfg),
        FixedSpreadStrategy(cfg, half_spread_ticks=1.0),
        AvellanedaStoikov(cfg, kappa=models["kappa"]),
        BayesianQuoter(cfg, models["queue"], models["markout"], models["cr"]),
    ]
    if rl_agent is not None:
        str50.append(RLStrategy(cfg, rl_agent))
    return str50


def run_backtests(cfg: dict, md_test: MarketData, F_test: np.ndarray, strategies: List,
                  max_duration_s: Optional[float] = None, verbose: bool = False) -> list:
    results = []
    for strat in strategies:
        # each strategy gets a fresh execution model over the same data
        if hasattr(strat, "reset"):
            strat.reset()
        if isinstance(strat, BayesianQuoter):
            # re-seed the online Bayesian posterior from the test stream start
            strat.qm = strat.qm  # keep fitted priors; online update adapts within-session
        eng = BacktestEngine(cfg, ReplayExecution(md_test, start_ts=1.0), strat, features=F_test,
                             max_duration_s=max_duration_s, verbose=verbose)
        r = eng.run()
        results.append(r)
        if verbose:
            s = summarize_result(r, md_test.tick_size)
            print(f"  {r.name:22s} net_pnl={s['net_pnl']:+.3f} obj={s['objective']:+.3f} "
                  f"fills={s['n_fills']} advsel={s['adverse_selection_1s']:.3f} [{r.wall_time_s:.1f}s]")
    return results


def run_full_pipeline(cfg: Optional[dict] = None, train_seed: int = 1, test_seed: int = 2,
                      train_duration_s: Optional[float] = None, test_duration_s: Optional[float] = None,
                      backtest_duration_s: Optional[float] = None, fit_hawkes: bool = True,
                      rl_agent=None, cache_dir: Optional[str] = None, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    seed_everything(int(cfg.get("seed", 42)))
    t_all = time.time()

    if verbose:
        print("[1/5] generating market data")
    md_tr = generate_synthetic(cfg, seed=train_seed, duration_s=train_duration_s, cache_dir=cache_dir)
    md_te = generate_synthetic(cfg, seed=test_seed, duration_s=test_duration_s, cache_dir=cache_dir)
    F_tr, F_te = compute_features(md_tr), compute_features(md_te)

    hawkes_info = {"params": md_tr.meta.get("hawkes"), "branching_ratio": HawkesParams.from_config(cfg["hawkes"]).branching_ratio()}
    if fit_hawkes and md_tr.meta.get("events") is not None and len(md_tr.meta["events"]):
        if verbose:
            print("[2/5] fitting Hawkes MLE")
        ev = md_tr.meta["events"]
        est = fit_hawkes_mle(ev[:, 0], ev[:, 1].astype(int), D=6, T=md_tr.book.ts[-1], beta_init=2.0, shared_beta=True)
        hawkes_info["fit"] = est.to_dict()
        hawkes_info["fit_branching_ratio"] = est.branching_ratio()

    if verbose:
        print("[3/5] fitting fill / markout / queue models")
    models = build_models(cfg, md_tr, F_tr, seed=int(cfg.get("seed", 42)), verbose=verbose)
    ds_tr = models["probes"]

    if verbose:
        print("[4/5] evaluating fill-probability model (out-of-sample probes)")
    ds_te = generate_probe_dataset(md_te, n_probes=min(8000, int(cfg["fill_model"]["n_probe_orders"])),
                                   max_wait_s=float(cfg["fill_model"]["max_wait_s"]),
                                   order_size=float(cfg["strategy"]["quote_size"]),
                                   horizons=tuple(cfg["strategy"]["markout_horizons_s"]), seed=test_seed + 100,
                                   features=F_te, lot=float(cfg["instrument"]["lot_size"]))
    fill_eval = {}
    for tau in cfg["fill_model"]["horizons_s"]:
        p = models["cr"].prob_fill_within(ds_te.X, float(tau))
        fill_eval[float(tau)] = evaluate_fill_model(p, ds_te.filled_within(float(tau)).astype(int))

    if verbose:
        print("[5/5] running backtests")
    strategies = make_strategies(cfg, models, rl_agent=rl_agent)
    results = run_backtests(cfg, md_te, F_te, strategies, max_duration_s=backtest_duration_s, verbose=verbose)

    bundle = {
        "config": dict(cfg),
        "wall_time_s": time.time() - t_all,
        "data": {"train": md_tr.summary(), "test": md_te.summary()},
        "hawkes": hawkes_info,
        "fill_eval": fill_eval,
        "aj_train": aalen_johansen(ds_tr.time, ds_tr.cause, np.linspace(0.05, float(cfg["fill_model"]["max_wait_s"]), 60)),
        "markout_table": markout_table(ds_tr, md_tr.tick_size),
        "markout_r2": models["markout"].r2_,
        "queue": {"rate_bid": models["queue"].rate_mean(1), "rate_ask": models["queue"].rate_mean(-1),
                  "m": models["queue"].m.tolist()},
        "kappa": models["kappa"],
        "results": results,
        "tick_size": md_te.tick_size,
        "symbol": md_te.symbol,
        "_models": models, "_md_test": md_te, "_F_test": F_te, "_ds_test": ds_te,
    }
    if verbose:
        print(f"pipeline done in {bundle['wall_time_s']:.1f}s")
    return bundle


# ---------------------------------------------------------------------------
def bundle_to_json(bundle: dict) -> dict:
    tick = bundle["tick_size"]
    out = {
        "symbol": bundle["symbol"], "tick_size": tick, "wall_time_s": bundle["wall_time_s"],
        "data": bundle["data"], "hawkes": bundle["hawkes"], "kappa": bundle["kappa"],
        "queue": bundle["queue"], "markout_r2": {str(k): v for k, v in bundle["markout_r2"].items()},
        "markout_table": bundle["markout_table"],
        "fill_eval": {str(k): {kk: vv for kk, vv in v.items()} for k, v in bundle["fill_eval"].items()},
        "aj": {k: np.asarray(v).tolist() for k, v in bundle["aj_train"].items()},
        "strategies": [], "objective_components": [],
    }
    for r in bundle["results"]:
        s = summarize_result(r, tick)
        out["strategies"].append(s)
        out["objective_components"].append({
            "strategy": r.name, "spread_revenue": s["spread_revenue"],
            "adverse_selection": s["adverse_selection_1s"], "fees": s["fees"],
            "inventory_cost": s["inventory_cost"], "objective": s["objective"], "net_pnl": s["net_pnl"],
        })
        ts = _downsample(r.ts - (r.ts[0] if len(r.ts) else 0))
        out.setdefault("equity", {})[r.name] = {
            "t": ts.tolist(),
            "pnl": _downsample(r.mtm_pnl).tolist(),
            "inventory": _downsample(r.inventory).tolist(),
        }
        # markout profile averaged across fills
        if r.n_fills:
            out.setdefault("markout_profile", {})[r.name] = {
                "h": r.markout_horizons.tolist(),
                "markout_ticks": (np.nanmean(r.markouts, axis=0) / tick).tolist(),
            }
    return out


def save_bundle(bundle: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = bundle_to_json(bundle)
    with open(out_dir / "dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, default=float)
    import pandas as pd
    df = pd.DataFrame(js["strategies"]).set_index("strategy")
    df.to_csv(out_dir / "comparison.csv")
    return out_dir / "dashboard_data.json"
