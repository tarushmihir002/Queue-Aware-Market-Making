import React from "react";
import { niceName } from "../theme.js";

const f = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : (+x).toFixed(d));
const sign = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");

export default function Kpis({ data }) {
  const strat = data.strategies || [];
  if (!strat.length) return null;
  const best = strat.reduce((a, b) => (b.objective > a.objective ? b : a));
  const bq = strat.find((s) => s.strategy === "bayesian_queue");
  const as_ = strat.find((s) => s.strategy === "avellaneda_stoikov");
  const tiles = [
    { l: "Best by objective", v: niceName(best.strategy), d: `net P&L ${f(best.net_pnl, 1)}` },
  ];
  if (bq) tiles.push({ l: "Bayesian net P&L", v: f(bq.net_pnl, 1), d: `objective ${f(bq.objective, 1)}`, c: sign(bq.net_pnl) });
  if (bq && as_) tiles.push({ l: "Bayesian − AS objective", v: `${bq.objective - as_.objective >= 0 ? "+" : ""}${f(bq.objective - as_.objective, 1)}`, d: "adverse-selection edge", c: bq.objective - as_.objective > 0 ? "pos" : "neg" });
  if (bq) tiles.push({ l: "Bayesian adverse sel.", v: f(bq.adverse_selection_1s, 2), d: "lowest = least toxic fills" });
  return (
    <div className="kpis">
      {tiles.map((t, i) => (
        <div className="kpi" key={i}>
          <div className="l">{t.l}</div>
          <div className={`v ${t.c || ""}`}>{t.v}</div>
          {t.d && <div className="d">{t.d}</div>}
        </div>
      ))}
    </div>
  );
}
