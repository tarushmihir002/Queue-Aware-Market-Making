#!/usr/bin/env python
"""
Interactive queue-aware market-making dashboard (Streamlit, live).

    streamlit run dashboard/streamlit_app.py

Tune the market regime (Hawkes order flow), the economics (fees, tick) and the
strategy parameters in the sidebar, then re-run the full pipeline and compare
the classical, Bayesian and (optional) RL quoters out-of-sample. Everything is
computed by the same qamm.report.run_full_pipeline used by the CLI, so numbers
match the paper exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from qamm import plotting
from qamm.backtest import summarize_result
from qamm.report import run_full_pipeline
from qamm.utils import load_config

st.set_page_config(page_title="Queue-Aware Market Making", layout="wide")


@st.cache_data(show_spinner=True)
def _run(cfg_overrides: tuple, train_dur: float, test_dur: float, seed_pair: tuple):
    cfg = load_config()
    for k, v in cfg_overrides:
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    bundle = run_full_pipeline(cfg, train_seed=seed_pair[0], test_seed=seed_pair[1],
                               train_duration_s=train_dur, test_duration_s=test_dur,
                               fit_hawkes=False, verbose=False)
    # strip unpicklable live objects for caching
    rows = [summarize_result(r, bundle["tick_size"]) for r in bundle["results"]]
    equity = {r.name: (r.ts - r.ts[0], r.mtm_pnl, r.inventory,
                       (np.nanmean(r.markouts, 0) / bundle["tick_size"] if r.n_fills else np.zeros(len(r.markout_horizons))),
                       r.markout_horizons) for r in bundle["results"]}
    light = {k: bundle[k] for k in ("data", "fill_eval", "aj_train", "markout_table", "markout_r2",
                                    "queue", "kappa", "tick_size", "symbol", "wall_time_s", "hawkes")}
    light["summaries"] = rows
    light["equity"] = equity
    return light


st.title("Queue-Aware Market Making — RL & Bayesian Fill Probability")
st.caption("Hawkes order flow · survival-analysis fill probability · adverse-selection markouts · "
           "Avellaneda–Stoikov vs Bayesian vs RL")

with st.sidebar:
    st.header("Market regime")
    mom = st.slider("Market-order self-excitation (momentum)", 0.0, 0.8, 0.35, 0.05)
    tox = st.slider("Cancel-after-trade coupling (toxicity)", 0.0, 0.8, 0.40, 0.05)
    st.header("Economics")
    maker = st.slider("Maker fee (bps)", 0.0, 2.0, 0.2, 0.1)
    st.header("Strategy")
    gamma = st.select_slider("AS / Bayesian risk aversion γ", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], 0.05)
    qsize = st.select_slider("Quote size", [0.5, 1.0, 2.0, 5.0], 1.0)
    st.header("Sim length")
    train_dur = st.select_slider("Train seconds", [600, 900, 1800, 3600], 900)
    test_dur = st.select_slider("Test seconds", [600, 900, 1800, 3600], 900)
    go = st.button("Run pipeline", type="primary")

overrides = (
    ("hawkes.alpha", None),  # placeholder; applied below
    ("fees.maker_bps", maker),
    ("strategy.gamma", gamma),
    ("strategy.quote_size", qsize),
)


def _apply_regime(cfg_overrides):
    cfg = load_config()
    a = np.array(cfg["hawkes"]["alpha"], float)
    a[0, 0] = a[1, 1] = mom
    a[4, 1] = a[5, 0] = tox
    return [("hawkes.alpha", a.tolist())] + [o for o in cfg_overrides if o[0] != "hawkes.alpha"]


if go or "light" not in st.session_state:
    with st.spinner("running Hawkes sim, fitting models, backtesting…"):
        st.session_state["light"] = _run(tuple(_apply_regime(overrides)), float(train_dur), float(test_dur), (1, 2))

light = st.session_state["light"]
tab1, tab2, tab3, tab4 = st.tabs(["📊 Strategies", "🎯 Fill model", "⚠️ Adverse selection", "📈 Data"])

with tab1:
    df = pd.DataFrame(light["summaries"]).set_index("strategy")
    show = ["net_pnl", "objective", "spread_revenue", "adverse_selection_1s", "fees", "inventory_cost",
            "n_fills", "fill_rate_per_quote", "avg_markout_ticks", "mean_abs_inventory", "sharpe_annualised"]
    best = df["objective"].idxmax()
    c1, c2, c3 = st.columns(3)
    c1.metric("Best strategy (objective)", best)
    c2.metric("Best net PnL", f"{df['net_pnl'].max():.3f}")
    c3.metric("Bayesian vs AS Δobjective",
              f"{df.loc['bayesian_queue','objective'] - df.loc.get('avellaneda_stoikov', df.iloc[0])['objective']:.3f}"
              if "bayesian_queue" in df.index else "n/a")
    st.dataframe(df[show].style.format("{:.4f}").highlight_max(subset=["net_pnl", "objective"], color="#204a2a"),
                 use_container_width=True)
    col = st.columns(2)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (t, pnl, inv, mo, h) in light["equity"].items():
        ax.plot(t, pnl, label=name, lw=1.3)
    ax.axhline(0, color="k", lw=0.6); ax.legend(fontsize=8); ax.set_xlabel("s"); ax.set_ylabel("P&L")
    ax.set_title("Out-of-sample equity"); col[0].pyplot(fig)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for name, (t, pnl, inv, mo, h) in light["equity"].items():
        ax2.plot(t, inv, label=name, lw=1.0, alpha=0.8)
    ax2.axhline(0, color="k", lw=0.6); ax2.legend(fontsize=8); ax2.set_xlabel("s"); ax2.set_ylabel("inventory")
    ax2.set_title("Inventory"); col[1].pyplot(fig2)

with tab2:
    st.subheader("Fill-probability model (competing risks)")
    cols = st.columns(len(light["fill_eval"]))
    for (tau, ev), c in zip(light["fill_eval"].items(), cols):
        c.metric(f"AUC @ {tau:g}s", f"{ev['auc']:.3f}", f"Brier {ev['brier']:.3f}")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    for tau, ev in light["fill_eval"].items():
        rel = np.array(ev["reliability"])
        if len(rel):
            ax.plot(rel[:, 0], rel[:, 1], "o-", label=f"τ={tau:g}s")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5); ax.legend(); ax.set_xlabel("predicted"); ax.set_ylabel("observed")
    ax.set_title("Calibration")
    aj = light["aj_train"]; fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(aj["grid"], aj["cif_fill"], label="fill", color="#2d8f5b")
    ax2.plot(aj["grid"], aj["cif_requote"], label="requote", color="#c23b3b")
    ax2.plot(aj["grid"], aj["surv"], label="resting", color="#444", ls=":"); ax2.legend()
    ax2.set_xlabel("s"); ax2.set_title("Cumulative incidence")
    a, b = st.columns(2); a.pyplot(fig); b.pyplot(fig2)

with tab3:
    st.subheader("Adverse selection vs queue position")
    mt = pd.DataFrame(light["markout_table"])
    st.dataframe(mt.round(4), use_container_width=True)
    import matplotlib.pyplot as plt
    sub = mt[mt.dist_ticks == 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(0.5 * (sub.queue_lo + sub.queue_hi.clip(upper=2)), sub.markout_ticks, yerr=sub.markout_se,
                fmt="o-", capsize=3)
    ax.axhline(0, color="k"); ax.set_xlabel("queue ratio"); ax.set_ylabel("markout @1s (ticks)")
    ax.set_title("Toxicity of fills by queue position"); st.pyplot(fig)

with tab4:
    st.subheader("Market data")
    st.json(light["data"])
    st.write("Queue-depletion posterior rate (events/s):", light["queue"])
    st.write("κ (fill-intensity decay / tick):", round(light["kappa"], 3))
    st.caption(f"pipeline wall time: {light['wall_time_s']:.1f}s")
