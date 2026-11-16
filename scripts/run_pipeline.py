#!/usr/bin/env python
"""
Reproducible end-to-end pipeline:
  generate (or load) market data -> fit Hawkes + fill/markout/queue models ->
  train/load the RL agent -> backtest every strategy -> save a results bundle +
  figures that the dashboards and the paper read.

Examples
--------
  python scripts/run_pipeline.py                         # synthetic, no RL, quick
  python scripts/run_pipeline.py --rl --train-rl         # include a freshly trained DQN
  python scripts/run_pipeline.py --config config/default.yaml --out results/run1
  python scripts/run_pipeline.py --binance BTCUSDT 2024-05-01 --window 0 1800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from qamm.models import compute_features
from qamm.report import run_full_pipeline, save_bundle
from qamm.utils import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "run"))
    ap.add_argument("--train-duration", type=float, default=1800.0)
    ap.add_argument("--test-duration", type=float, default=1800.0)
    ap.add_argument("--backtest-duration", type=float, default=None, help="cap backtest length (s)")
    ap.add_argument("--train-seed", type=int, default=1)
    ap.add_argument("--test-seed", type=int, default=2)
    ap.add_argument("--no-hawkes-fit", action="store_true")
    ap.add_argument("--rl", action="store_true", help="include an RL strategy in the comparison")
    ap.add_argument("--train-rl", action="store_true", help="train a new DQN (else load results/agent.pt)")
    ap.add_argument("--rl-episodes", type=int, default=None)
    ap.add_argument("--binance", nargs=2, metavar=("SYMBOL", "DAY"), default=None,
                    help="use a downloaded Binance day instead of synthetic data for the TEST set")
    ap.add_argument("--window", nargs=2, type=float, metavar=("START_S", "DURATION_S"), default=None)
    ap.add_argument("--plots", action="store_true", help="also render matplotlib figures")
    ap.add_argument("--make-dashboard", action="store_true", help="write the self-contained HTML dashboard")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    cache = ROOT / "data" / "cache"

    rl_agent = None
    if args.rl:
        rl_agent = _get_rl_agent(cfg, args, cache)

    # optional: real Binance test data
    md_test_override = F_test_override = None
    if args.binance:
        from qamm.data import load_binance_day
        sym, day = args.binance
        start, dur = (args.window if args.window else (0.0, args.test_duration))
        print(f"loading Binance {sym} {day} window [{start}, {start+dur}]s")
        md_test_override = load_binance_day(sym, day, data_dir=str(ROOT / "data" / "binance"),
                                            start_s=start, duration_s=dur)
        F_test_override = compute_features(md_test_override)

    if md_test_override is None:
        bundle = run_full_pipeline(cfg, train_seed=args.train_seed, test_seed=args.test_seed,
                                   train_duration_s=args.train_duration, test_duration_s=args.test_duration,
                                   backtest_duration_s=args.backtest_duration,
                                   fit_hawkes=not args.no_hawkes_fit, rl_agent=rl_agent, cache_dir=str(cache))
    else:
        bundle = _pipeline_with_real_test(cfg, args, md_test_override, F_test_override, rl_agent, cache)

    path = save_bundle(bundle, out)
    print(f"\nsaved results -> {path}")
    import pandas as pd
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    from qamm.backtest import compare_results
    df = compare_results(bundle["results"], bundle["tick_size"])
    cols = ["net_pnl", "objective", "spread_revenue", "adverse_selection_1s", "fees", "inventory_cost",
            "n_fills", "fill_rate_per_quote", "avg_markout_ticks", "mean_abs_inventory", "sharpe_annualised"]
    print("\n==== STRATEGY COMPARISON (out-of-sample) ====")
    print(df[cols].round(4).to_string())
    best = df["objective"].idxmax()
    print(f"\nbest objective: {best}  (net PnL {df.loc[best,'net_pnl']:.3f})")

    if args.plots:
        from qamm.plotting import render_all
        figdir = render_all(bundle, out / "figures")
        print(f"figures -> {figdir}")
    if args.make_dashboard:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dashboard.py"),
                        "--data", str(out / "dashboard_data.json"), "--out", str(out / "dashboard.html")])


def _get_rl_agent(cfg, args, cache):
    from qamm.data import generate_synthetic
    from qamm.strategy import DQNAgent, MarketMakingEnv, train_dqn
    agent_path = ROOT / "results" / "agent.pt"
    if not args.train_rl and agent_path.exists():
        print(f"loading RL agent <- {agent_path}")
        return DQNAgent.load(agent_path, seed=0)
    print("training RL agent (DQN)")
    mds = [generate_synthetic(cfg, seed=s, duration_s=args.train_duration, cache_dir=str(cache)) for s in (1, 3, 4)]
    Fs = [compute_features(m) for m in mds]
    rc = cfg["rl"]
    env = MarketMakingEnv(cfg, mds, Fs, episode_len_steps=int(rc["episode_len_steps"]), seed=0)
    ag = DQNAgent(lr=float(rc["lr"]), gamma=float(rc["gamma"]), batch_size=int(rc["batch_size"]),
                  buffer_size=int(rc["buffer_size"]), target_update=int(rc["target_update"]),
                  eps_start=float(rc["epsilon_start"]), eps_end=float(rc["epsilon_end"]),
                  eps_decay_steps=int(rc["epsilon_decay_steps"]), hidden=int(rc["hidden"]), seed=0)
    train_dqn(env, ag, episodes=int(args.rl_episodes or rc["episodes"]), warmup_episodes=2)
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    ag.save(agent_path)
    print(f"saved RL agent -> {agent_path}")
    return ag


def _pipeline_with_real_test(cfg, args, md_te, F_te, rl_agent, cache):
    """Train models on synthetic data, evaluate strategies on the real Binance window."""
    from qamm.data import generate_synthetic
    from qamm.report import build_models, make_strategies, run_backtests
    from qamm.backtest import summarize_result
    from qamm.models import aalen_johansen, evaluate_fill_model, generate_probe_dataset, markout_table
    md_tr = generate_synthetic(cfg, seed=args.train_seed, duration_s=args.train_duration, cache_dir=str(cache))
    F_tr = compute_features(md_tr)
    models = build_models(cfg, md_tr, F_tr, seed=int(cfg.get("seed", 42)))
    ds_tr = models["probes"]
    strategies = make_strategies(cfg, models, rl_agent=rl_agent)
    results = run_backtests(cfg, md_te, F_te, strategies, max_duration_s=args.backtest_duration, verbose=True)
    ds_te = generate_probe_dataset(md_te, n_probes=6000, max_wait_s=float(cfg["fill_model"]["max_wait_s"]),
                                   order_size=float(cfg["strategy"]["quote_size"]),
                                   horizons=tuple(cfg["strategy"]["markout_horizons_s"]), seed=7, features=F_te,
                                   lot=float(cfg["instrument"]["lot_size"]))
    fill_eval = {}
    for tau in cfg["fill_model"]["horizons_s"]:
        p = models["cr"].prob_fill_within(ds_te.X, float(tau))
        fill_eval[float(tau)] = evaluate_fill_model(p, ds_te.filled_within(float(tau)).astype(int))
    return {
        "config": dict(cfg), "wall_time_s": 0.0,
        "data": {"train": md_tr.summary(), "test": md_te.summary()},
        "hawkes": {"params": md_tr.meta.get("hawkes")}, "kappa": models["kappa"],
        "fill_eval": fill_eval,
        "aj_train": aalen_johansen(ds_tr.time, ds_tr.cause, np.linspace(0.05, 10, 60)),
        "markout_table": markout_table(ds_tr, md_tr.tick_size), "markout_r2": models["markout"].r2_,
        "queue": {"rate_bid": models["queue"].rate_mean(1), "rate_ask": models["queue"].rate_mean(-1),
                  "m": models["queue"].m.tolist()},
        "results": results, "tick_size": md_te.tick_size, "symbol": md_te.symbol,
        "_models": models, "_md_test": md_te, "_F_test": F_te, "_ds_test": ds_te,
    }


if __name__ == "__main__":
    main()
