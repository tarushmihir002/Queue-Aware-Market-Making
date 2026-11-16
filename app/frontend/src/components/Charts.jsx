import React from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ScatterChart, Scatter, BarChart, Bar, Cell, ReferenceLine,
} from "recharts";
import { STRAT_COLORS, SERIES, niceName } from "../theme.js";

const H = 300;

function tip(tokens) {
  return {
    contentStyle: { background: tokens.panel || "var(--panel)", border: `1px solid ${tokens.line}`,
      borderRadius: 10, fontSize: 12, color: tokens.ink },
    labelStyle: { color: tokens.muted }, itemStyle: { color: tokens.ink },
  };
}
const axis = (tokens) => ({ stroke: tokens.line, tick: { fill: tokens.muted, fontSize: 11 } });

// equity / inventory: strategies share the downsampled time grid, so merge by index
function mergeEquity(equity, field) {
  const names = Object.keys(equity);
  if (!names.length) return { data: [], names };
  const t = equity[names[0]].t;
  const data = t.map((tv, i) => {
    const row = { t: Math.round(tv) };
    names.forEach((n) => { row[n] = equity[n][field][i]; });
    return row;
  });
  return { data, names };
}

export function LineSeries({ equity, field, tokens, ylabel }) {
  const { data, names } = mergeEquity(equity, field);
  return (
    <ResponsiveContainer width="100%" height={H}>
      <LineChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 2 }}>
        <CartesianGrid stroke={tokens.line} strokeDasharray="2 3" />
        <XAxis dataKey="t" {...axis(tokens)} tickFormatter={(v) => `${v}s`} />
        <YAxis {...axis(tokens)} width={48} />
        <Tooltip {...tip(tokens)} formatter={(v, n) => [Number(v).toFixed(3), niceName(n)]}
                 labelFormatter={(v) => `t = ${v}s`} />
        <Legend formatter={(v) => niceName(v)} wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine y={0} stroke={tokens.muted} strokeWidth={1} />
        {names.map((n) => (
          <Line key={n} type="monotone" dataKey={n} dot={false} strokeWidth={1.8}
                stroke={STRAT_COLORS[n] || SERIES[0]} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function Calibration({ fillEval, tokens }) {
  const taus = Object.keys(fillEval);
  return (
    <ResponsiveContainer width="100%" height={H}>
      <ScatterChart margin={{ top: 6, right: 12, bottom: 18, left: 2 }}>
        <CartesianGrid stroke={tokens.line} strokeDasharray="2 3" />
        <XAxis type="number" dataKey="x" domain={[0, 1]} {...axis(tokens)}
               label={{ value: "predicted P(fill ≤ τ)", position: "insideBottom", offset: -8, fill: tokens.muted, fontSize: 11 }} />
        <YAxis type="number" dataKey="y" domain={[0, 1]} width={40} {...axis(tokens)} />
        <Tooltip {...tip(tokens)} formatter={(v) => Number(v).toFixed(3)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={tokens.muted} strokeDasharray="4 4" ifOverflow="hidden" />
        {taus.map((t, i) => (
          <Scatter key={t} name={`τ=${t}s (AUC ${(+fillEval[t].auc).toFixed(2)})`}
                   data={fillEval[t].reliability.map((r) => ({ x: r[0], y: r[1] }))}
                   fill={SERIES[i % SERIES.length]} line={{ stroke: SERIES[i % SERIES.length] }}
                   isAnimationActive={false} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function Cif({ aj, tokens }) {
  const data = aj.grid.map((g, i) => ({
    t: +g.toFixed(2), filled: aj.cif_fill[i], requoted: aj.cif_requote[i], resting: aj.surv[i],
  }));
  const defs = [["filled", "var(--green)", "#0e9f70"], ["requoted", "var(--red)", "#d94759"],
                ["resting", "var(--muted)", "#8a97b0"]];
  return (
    <ResponsiveContainer width="100%" height={H}>
      <LineChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: 2 }}>
        <CartesianGrid stroke={tokens.line} strokeDasharray="2 3" />
        <XAxis dataKey="t" {...axis(tokens)} tickFormatter={(v) => `${v}s`} />
        <YAxis domain={[0, 1]} width={40} {...axis(tokens)} />
        <Tooltip {...tip(tokens)} formatter={(v) => Number(v).toFixed(3)} labelFormatter={(v) => `${v}s`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {defs.map(([k, , c]) => (
          <Line key={k} type="monotone" dataKey={k} dot={false} strokeWidth={1.8} stroke={c} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ValueBars({ data, xKey, yKey, tokens, colorBySign = false, fmt = (v) => v.toFixed(2) }) {
  return (
    <ResponsiveContainer width="100%" height={H}>
      <BarChart data={data} margin={{ top: 6, right: 12, bottom: 44, left: 2 }}>
        <CartesianGrid stroke={tokens.line} strokeDasharray="2 3" vertical={false} />
        <XAxis dataKey={xKey} {...axis(tokens)} angle={-28} textAnchor="end" interval={0} height={56} />
        <YAxis {...axis(tokens)} width={46} />
        <Tooltip {...tip(tokens)} formatter={(v) => fmt(Number(v))} />
        <ReferenceLine y={0} stroke={tokens.muted} />
        <Bar dataKey={yKey} radius={[3, 3, 0, 0]} isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={colorBySign ? (d[yKey] < 0 ? "#d94759" : "#0e9f70") : "#2d77d4"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
