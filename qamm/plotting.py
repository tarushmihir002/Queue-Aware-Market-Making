"""Matplotlib figures for the paper / CLI (saved as PNG). The interactive
dashboards use their own rendering; this is for the static report."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .backtest import summarize_result


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_fill_calibration(bundle, ax=None):
    plt = _mpl()
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5, 5))
    for tau, ev in bundle["fill_eval"].items():
        rel = np.array(ev["reliability"])
        if len(rel):
            ax.plot(rel[:, 0], rel[:, 1], "o-", label=f"τ={tau:g}s (AUC {ev['auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("predicted P(fill ≤ τ)"); ax.set_ylabel("observed fill frequency")
    ax.set_title("Fill-probability calibration"); ax.legend(fontsize=8)
    return ax


def fig_cif(bundle, ax=None):
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    aj = bundle["aj_train"]
    g = np.asarray(aj["grid"])
    ax.plot(g, aj["cif_fill"], label="CIF fill", color="#2d8f5b")
    ax.plot(g, aj["cif_requote"], label="CIF requote (cancel)", color="#c23b3b")
    ax.plot(g, aj["surv"], label="still resting", color="#444", ls=":")
    ax.set_xlabel("time since placement (s)"); ax.set_ylabel("cumulative incidence")
    ax.set_title("Competing risks (Aalen–Johansen)"); ax.legend(fontsize=8)
    return ax


def fig_markout_vs_queue(bundle, ax=None):
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    rows = [r for r in bundle["markout_table"] if r["dist_ticks"] == 0]
    x = [0.5 * (r["queue_lo"] + min(r["queue_hi"], 2)) for r in rows]
    y = [r["markout_ticks"] for r in rows]
    e = [r["markout_se"] for r in rows]
    ax.errorbar(x, y, yerr=e, fmt="o-", color="#2b5fa8", capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("queue ratio (volume ahead / top size)"); ax.set_ylabel("markout @1s (ticks)")
    ax.set_title("Adverse selection vs queue position (at the touch)")
    return ax


def fig_equity(bundle, ax=None):
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    for r in bundle["results"]:
        ax.plot(r.ts - r.ts[0], r.mtm_pnl, label=r.name, lw=1.3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (s)"); ax.set_ylabel("mark-to-mid P&L (quote ccy)")
    ax.set_title("Out-of-sample equity curves"); ax.legend(fontsize=8)
    return ax


def fig_inventory(bundle, ax=None):
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.2))
    for r in bundle["results"]:
        ax.plot(r.ts - r.ts[0], r.inventory, label=r.name, lw=1.0, alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (s)"); ax.set_ylabel("inventory (base units)")
    ax.set_title("Inventory trajectories"); ax.legend(fontsize=8)
    return ax


def fig_objective_decomposition(bundle, ax=None):
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    tick = bundle["tick_size"]
    names, spread, adv, fees, invc = [], [], [], [], []
    for r in bundle["results"]:
        s = summarize_result(r, tick)
        names.append(r.name); spread.append(s["spread_revenue"]); adv.append(-s["adverse_selection_1s"])
        fees.append(-s["fees"]); invc.append(-s["inventory_cost"])
    x = np.arange(len(names))
    ax.bar(x, spread, label="spread revenue", color="#2d8f5b")
    bottom = np.array(spread)
    ax.bar(x, adv, bottom=bottom, label="− adverse selection", color="#c23b3b")
    bottom = bottom + np.array(adv)
    ax.bar(x, fees, bottom=bottom, label="− fees", color="#e0953b")
    bottom = bottom + np.array(fees)
    ax.bar(x, invc, bottom=bottom, label="− inventory cost", color="#7a5bbf")
    obj = [summarize_result(r, tick)["objective"] for r in bundle["results"]]
    ax.plot(x, obj, "ko-", label="objective (net)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("quote ccy"); ax.set_title("Objective decomposition")
    ax.legend(fontsize=8)
    return ax


def render_all(bundle, out_dir: str | Path) -> Path:
    plt = _mpl()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in [("fill_calibration", fig_fill_calibration), ("competing_risks", fig_cif),
                     ("markout_vs_queue", fig_markout_vs_queue), ("equity", fig_equity),
                     ("inventory", fig_inventory), ("objective_decomposition", fig_objective_decomposition)]:
        ax = fn(bundle)
        ax.figure.tight_layout()
        ax.figure.savefig(out_dir / f"{name}.png", dpi=130)
        plt.close(ax.figure)
    return out_dir
