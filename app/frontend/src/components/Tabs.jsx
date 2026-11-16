import React from "react";
import { LineSeries, Calibration, Cif, ValueBars } from "./Charts.jsx";
import { niceName } from "../theme.js";

const f = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : (+x).toFixed(d));
const sign = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");

export const TAB_NAMES = ["Strategies", "Fill model", "Adverse selection", "Objective"];

function StrategyTable({ strat }) {
  const cols = [
    ["strategy", "Strategy"], ["net_pnl", "Net P&L", 2], ["objective", "Objective", 2],
    ["spread_revenue", "Spread", 2], ["adverse_selection_1s", "Adv sel", 2], ["fees", "Fees", 2],
    ["inventory_cost", "Inv cost", 3], ["n_fills", "Fills", 0], ["fill_rate_per_quote", "Fill/quote", 2],
    ["avg_markout_ticks", "Markout(t)", 3], ["mean_abs_inventory", "|inv|", 1], ["sharpe_annualised", "Sharpe", 0],
  ];
  const best = strat.reduce((a, b) => (b.objective > a.objective ? b : a)).strategy;
  return (
    <div className="card scroll">
      <table>
        <thead><tr>{cols.map((c) => <th key={c[0]}>{c[1]}</th>)}</tr></thead>
        <tbody>
          {strat.map((s) => (
            <tr key={s.strategy} className={s.strategy === best ? "best" : ""}>
              {cols.map((c) => {
                if (c[0] === "strategy") return <td key={c[0]}>{niceName(s.strategy)}</td>;
                const cls = ["net_pnl", "objective", "spread_revenue"].includes(c[0]) ? sign(s[c[0]]) : "";
                return <td key={c[0]} className={cls}>{c[2] === 0 ? s[c[0]] : f(s[c[0]], c[2])}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Tabs({ tab, data, tokens }) {
  if (tab === 0) {
    return (
      <>
        <StrategyTable strat={data.strategies} />
        <div className="grid2" style={{ marginTop: 14 }}>
          <div className="card">
            <h2>Out-of-sample equity (P&L)</h2>
            <p className="cap">Mark-to-mid P&L over the held-out test window.</p>
            <LineSeries equity={data.equity} field="pnl" tokens={tokens} />
          </div>
          <div className="card">
            <h2>Inventory trajectory</h2>
            <p className="cap">Base units held; near zero = well-hedged.</p>
            <LineSeries equity={data.equity} field="inventory" tokens={tokens} />
          </div>
        </div>
      </>
    );
  }
  if (tab === 1) {
    const fe = data.fill_eval || {};
    return (
      <>
        <div className="kpis">
          {Object.keys(fe).map((t) => (
            <div className="kpi" key={t}>
              <div className="l">AUC @ {t}s</div>
              <div className="v">{f(fe[t].auc, 3)}</div>
              <div className="d">Brier {f(fe[t].brier, 3)} vs {f(fe[t].brier_const, 3)} base</div>
            </div>
          ))}
        </div>
        <div className="grid2">
          <div className="card">
            <h2>Fill-probability calibration</h2>
            <p className="cap">Points on the diagonal = well-calibrated probabilities.</p>
            <Calibration fillEval={fe} tokens={tokens} />
          </div>
          <div className="card">
            <h2>Competing risks (Aalen–Johansen)</h2>
            <p className="cap">Fill competes with cancellation; the gap is pure adverse-selection exposure.</p>
            <Cif aj={data.aj} tokens={tokens} />
          </div>
        </div>
      </>
    );
  }
  if (tab === 2) {
    const touch = (data.markout_table || []).filter((r) => r.dist_ticks === 0);
    const bars = touch.map((r) => ({
      bucket: `q ${r.queue_lo}–${r.queue_hi >= 10 ? "∞" : r.queue_hi}`,
      markout: r.markout_ticks, fill: r.fill_rate,
    }));
    const cols = [["dist_ticks", "dist(t)", 0], ["q", "queue"], ["n", "n", 0], ["fill_rate", "fill rate", 2],
      ["p_fill_100ms", "P≤100ms", 2], ["p_fill_500ms", "P≤500ms", 2], ["p_fill_1s", "P≤1s", 2],
      ["p_requote", "P requote", 2], ["median_fill_time", "med fill(s)", 2], ["markout_ticks", "markout(t)", 2]];
    return (
      <>
        <div className="grid2">
          <div className="card">
            <h2>Markout @1s by queue position</h2>
            <p className="cap">At the touch. More volume ahead ⇒ slower, more adversely-selected fills.</p>
            <ValueBars data={bars} xKey="bucket" yKey="markout" tokens={tokens} colorBySign fmt={(v) => v.toFixed(3)} />
          </div>
          <div className="card">
            <h2>Fill rate by queue position</h2>
            <p className="cap">Probability of ever filling within the wait window.</p>
            <ValueBars data={bars} xKey="bucket" yKey="fill" tokens={tokens} fmt={(v) => v.toFixed(2)} />
          </div>
        </div>
        <div className="card scroll" style={{ marginTop: 14 }}>
          <table>
            <thead><tr>{cols.map((c) => <th key={c[0]}>{c[1]}</th>)}</tr></thead>
            <tbody>
              {(data.markout_table || []).map((r, i) => (
                <tr key={i}>
                  {cols.map((c) => {
                    if (c[0] === "q") return <td key={c[0]}>{r.queue_lo}–{r.queue_hi >= 10 ? "∞" : r.queue_hi}</td>;
                    const cls = c[0] === "markout_ticks" ? sign(r[c[0]]) : "";
                    return <td key={c[0]} className={cls}>{c[2] === 0 ? r[c[0]] : f(r[c[0]], c[2])}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    );
  }
  // objective
  const oc = data.objective_components || [];
  const best = oc.length ? Math.max(...oc.map((o) => o.objective)) : 0;
  const bars = oc.map((o) => ({ strategy: niceName(o.strategy), objective: o.objective }));
  return (
    <div className="card">
      <h2>Net objective by strategy</h2>
      <p className="cap">Objective = spread revenue − adverse selection − fees − inventory cost.</p>
      <ValueBars data={bars} xKey="strategy" yKey="objective" tokens={tokens} colorBySign />
      <div className="scroll" style={{ marginTop: 12 }}>
        <table>
          <thead><tr>{["Strategy", "Spread rev", "− Adv sel", "− Fees", "− Inv cost", "= Objective", "Net P&L"].map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {oc.map((o) => (
              <tr key={o.strategy} className={o.objective === best ? "best" : ""}>
                <td>{niceName(o.strategy)}</td>
                <td className="pos">{f(o.spread_revenue, 2)}</td>
                <td className="neg">{f(-o.adverse_selection, 2)}</td>
                <td className="neg">{f(-o.fees, 2)}</td>
                <td className="neg">{f(-o.inventory_cost, 3)}</td>
                <td className={sign(o.objective)}>{f(o.objective, 2)}</td>
                <td className={sign(o.net_pnl)}>{f(o.net_pnl, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
