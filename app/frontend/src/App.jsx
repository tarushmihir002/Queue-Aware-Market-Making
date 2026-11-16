import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import { useTheme, useTokens } from "./theme.js";
import Controls from "./components/Controls.jsx";
import Kpis from "./components/Kpis.jsx";
import Tabs, { TAB_NAMES } from "./components/Tabs.jsx";

const f = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : (+x).toFixed(d));

export default function App() {
  const [theme, cycleTheme] = useTheme();
  const tokens = useTokens(theme);
  const [params, setParams] = useState(null);
  const [data, setData] = useState(null);
  const [tab, setTab] = useState(0);
  const [status, setStatus] = useState({ running: false, has_rl_agent: false, progress: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  // bootstrap: config defaults + any precomputed results
  useEffect(() => {
    (async () => {
      try {
        const cfg = await api.config();
        setParams(cfg.defaults);
        const st = await api.status();
        setStatus(st);
        const r = await api.results();
        if (!r.empty) setData(r);
      } catch (e) {
        setError("Backend not reachable. Start it with: uvicorn app.backend.main:app --port 8000");
      }
    })();
  }, []);

  const run = async () => {
    setError(null);
    setBusy(true);
    const poll = setInterval(async () => {
      try { setStatus(await api.status()); } catch {}
    }, 1500);
    try {
      const r = await api.run(params);
      setData(r);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      clearInterval(poll);
      setBusy(false);
      try { setStatus(await api.status()); } catch {}
    }
  };

  const ctlStatus = { ...status, running: busy };

  const themeLabel = { system: "◐ System", light: "☀ Light", dark: "☾ Dark" }[theme];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          Queue-Aware MM
          <small>Bayesian fill prob · Hawkes flow · RL</small>
        </div>
        {params && <Controls params={params} setParams={setParams} onRun={run} status={ctlStatus} />}
        <div className={`status-line ${error ? "err" : ""}`}>
          {error ? error : busy ? `${status.progress || "running"}…` : data ? "Ready." : "Loading…"}
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <h1>Market-Making Dashboard</h1>
          <button className="theme-btn" onClick={cycleTheme}>{themeLabel}</button>
        </div>

        {!data && !error && <p className="sub">Loading results… if none exist yet, set parameters and press “Run backtest”.</p>}
        {error && <p className="sub">Could not load the backend — see the message in the sidebar.</p>}

        {data && (
          <>
            <p className="sub">
              {data.symbol} · tick {data.tick_size} · out-of-sample test{" "}
              {f((data.data?.test?.duration_s || 0) / 60, 0)} min ·{" "}
              {(data.data?.test?.n_book_updates || 0).toLocaleString()} book updates,{" "}
              {(data.data?.test?.n_trades || 0).toLocaleString()} trades.
            </p>
            <div className="badges">
              {data.hawkes && <Badge l="branching ratio" v={f(data.hawkes.branching_ratio, 2)} />}
              <Badge l="κ decay /tick" v={f(data.kappa, 2)} />
              {data.queue && <Badge l="queue rate bid" v={f(data.queue.rate_bid, 1) + "/s"} />}
              {data.markout_r2 && <Badge l="markout R² @1s" v={f(data.markout_r2["1.0"], 2)} />}
              {data.params_used && <Badge l="γ" v={f(data.params_used.gamma, 2)} />}
              {data.params_used && <Badge l="maker fee" v={f(data.params_used.maker_bps, 1) + " bp"} />}
            </div>

            <Kpis data={data} />

            <div className="tabs">
              {TAB_NAMES.map((n, i) => (
                <div key={n} className={`tab ${i === tab ? "on" : ""}`} onClick={() => setTab(i)}>{n}</div>
              ))}
            </div>
            <Tabs tab={tab} data={data} tokens={tokens} />

            <p className="foot">
              Objective = spread revenue − adverse selection − fees − inventory cost. P&L in quote currency;
              markouts in ticks. Computed by the qamm pipeline on this machine.
            </p>
          </>
        )}
      </main>

      {busy && (
        <div className="overlay">
          <div className="box">
            <div className="spinner" style={{ margin: "0 auto" }} />
            <p>Running backtest…<br />{status.progress}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function Badge({ l, v }) {
  return <span className="badge">{l} <b>{v}</b></span>;
}
