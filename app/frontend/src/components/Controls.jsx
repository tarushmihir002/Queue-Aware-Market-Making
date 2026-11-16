import React from "react";

function Slider({ label, unit, value, min, max, step, onChange }) {
  return (
    <div className="ctl">
      <label>
        <span>{label}</span>
        <b>{(+value).toFixed(step < 0.01 ? 3 : step < 1 ? 2 : 0)}{unit || ""}</b>
      </label>
      <input type="range" min={min} max={max} step={step} value={value}
             onChange={(e) => onChange(parseFloat(e.target.value))} />
    </div>
  );
}

export default function Controls({ params, setParams, onRun, status }) {
  const set = (k) => (v) => setParams((p) => ({ ...p, [k]: v }));
  const running = status.running;
  return (
    <>
      <div className="ctl-group">
        <h3>Strategy</h3>
        <Slider label="Risk aversion γ" value={params.gamma} min={0.01} max={0.5} step={0.01} onChange={set("gamma")} />
        <Slider label="Quote size" value={params.quote_size} min={0.5} max={10} step={0.5} onChange={set("quote_size")} />
      </div>
      <div className="ctl-group">
        <h3>Economics</h3>
        <Slider label="Maker fee" unit=" bp" value={params.maker_bps} min={0} max={2} step={0.1} onChange={set("maker_bps")} />
      </div>
      <div className="ctl-group">
        <h3>Market regime (Hawkes)</h3>
        <Slider label="Momentum" value={params.momentum} min={0} max={0.7} step={0.05} onChange={set("momentum")} />
        <Slider label="Toxicity" value={params.toxicity} min={0} max={0.7} step={0.05} onChange={set("toxicity")} />
        <p className="hint">Toxicity = how strongly trades trigger same-side cancels (drives adverse selection).</p>
      </div>
      <div className="ctl-group">
        <h3>Simulation</h3>
        <div className="ctl">
          <label><span>Train / test length</span></label>
          <select value={params.train_s}
                  onChange={(e) => setParams((p) => ({ ...p, train_s: +e.target.value, test_s: +e.target.value }))}>
            <option value={600}>10 min (fast)</option>
            <option value={900}>15 min</option>
            <option value={1800}>30 min</option>
            <option value={3600}>60 min (slow)</option>
          </select>
        </div>
        <div className="ctl row">
          <input id="rl" type="checkbox" checked={!!params.include_rl} disabled={!status.has_rl_agent}
                 onChange={(e) => setParams((p) => ({ ...p, include_rl: e.target.checked }))} />
          <label htmlFor="rl" style={{ margin: 0 }}>
            Include RL agent {status.has_rl_agent ? "" : "(train first)"}
          </label>
        </div>
      </div>
      <button className="run-btn" onClick={onRun} disabled={running}>
        {running ? "Running…" : "Run backtest"}
      </button>
    </>
  );
}
