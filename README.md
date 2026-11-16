# Queue-Aware Market Making with Bayesian Fill Probability, Hawkes Order Flow & RL

A research framework that answers the project's core question:

> Can we predict the probability that a passive order is filled, estimate the
> adverse-selection cost conditional on a fill, and choose the price and size at
> which to quote so that **net P&L after fees and adverse selection** beats a
> well-defined baseline?

It ships a reproducible pipeline — synthetic *and* real (Binance) Level-2 data,
a queue-tracking order-book simulator, survival-analysis fill models, Hawkes
order-flow estimation, adverse-selection markouts, an Avellaneda–Stoikov
baseline, and **Bayesian** and **reinforcement-learning** quoting policies —
plus a backtesting engine, a Streamlit dashboard and a self-contained HTML
dashboard.

---

## Headline result

On out-of-sample data the execution-aware **Bayesian queue** policy produces the
highest net P&L and the only positive *objective*
(`spread revenue − adverse selection − fees − inventory cost`), while carrying
the smallest inventory. The naive Avellaneda–Stoikov and fixed-spread baselines
are adversely selected and lose money; joining the touch earns spread but
accumulates dangerous inventory. Run `python scripts/run_pipeline.py` to
reproduce the exact table (see [`paper/research_paper.md`](paper/research_paper.md)).

---

## Install

```bash
python -m pip install -r requirements.txt     # or: pip install -e .[dashboard,rl,dev]
```

Python ≥ 3.10. Core: NumPy, SciPy, pandas, Polars, PyTorch, scikit-learn, Numba.
Optional: Streamlit (dashboard), gymnasium (RL env), pytest (tests).

## Quick start

```bash
# 1. full pipeline on synthetic data: data -> models -> backtests -> results/run/
python scripts/run_pipeline.py --plots

# 2. include a trained RL agent in the comparison
python scripts/run_pipeline.py --rl --train-rl --plots --make-dashboard

# 3a. React dashboard app on your own local server (FastAPI + React/Vite)
python scripts/serve_dashboard.py        # builds the frontend, serves API+app at http://127.0.0.1:8000

# 3b. interactive Streamlit dashboard (live re-runs as you move the sliders)
streamlit run dashboard/streamlit_app.py

# 4. static shareable dashboards (offline, self-contained)
python scripts/make_dashboard.py --data results/run/dashboard_data.json --out results/run/dashboard.html
python scripts/make_artifact.py  --data results/run/dashboard_data.json --out results/run/artifact_dashboard.html
```

`make_dashboard.py` writes a standalone file you can open directly;
`make_artifact.py` writes a claude.ai-artifact-ready page (publishable as a
shareable Claude artifact).

### Using real Binance Level-2 data

Public archive, no API key (https://data.binance.vision):

```bash
python scripts/download_binance.py BTCUSDT 2024-05-01 --window 0 3600 --save data/btc_1h.npz
python scripts/run_pipeline.py --binance BTCUSDT 2024-05-01 --window 0 3600

python scripts/run_pipeline.py --config config/default.yaml --out results/run --train-duration 1800 --test-duration 1800 --train-seed 1 --test-seed 2 --rl --train-rl --rl-episodes 60 --plots --make-dashboard

```

`bookTicker` gives exact top-of-book size (queue) and `aggTrades` every trade —
all the fill model and markout estimator need. Deeper multi-level history needs
a vendor feed (Tardis, crypto-lob-stream); drop it in via
`qamm.data.load_l2_csv(book_csv, trades_csv)` — the models are data-source
agnostic. NSE depth exports load the same way once renamed to the canonical
columns (`ts, bid_px_1.., bid_sz_1.., ask_px_1.., ask_sz_1..`).

## Local dashboard app (React + FastAPI)

A self-hosted web app under `app/`. The FastAPI backend (`app/backend`) exposes the
pipeline and serves the built React/Vite frontend (`app/frontend`), so one command
hosts everything at `http://127.0.0.1:8000`:

```bash
pip install fastapi "uvicorn[standard]"          # backend deps (also in requirements.txt)
python scripts/serve_dashboard.py                # npm install + build + serve
```

Move the sidebar controls (risk aversion γ, maker fee, market momentum/toxicity,
quote size, sim length) and press **Run backtest**: the backend re-runs the full
pipeline with those parameters and the dashboard redraws (strategies, fill model,
adverse selection, objective). API: `GET /api/results`, `POST /api/run`,
`GET /api/status|config|health`.

For live frontend development:
```bash
uvicorn app.backend.main:app --port 8000         # terminal 1
cd app/frontend && npm install && npm run dev     # terminal 2 (Vite @ :5173, proxies /api)
```

## Architecture

```
qamm/
  hawkes/        multivariate Hawkes: Ogata simulation, exp-kernel MLE, branching ratio
  lob/           queue-tracking order book + 6-type Hawkes-driven event simulator
  data/          canonical schema, synthetic generator, Binance & CSV loaders
  models/
    features.py           causal market-state features (batch + online)
    survival.py           competing-risks fill model (Aalen–Johansen, discrete-time NN,
                          closed-form Bayesian queue model)
    adverse_selection.py  markout (adverse-selection) regression + queue-position table
  strategy/
    avellaneda_stoikov.py classical AS market maker (κ calibrated from fills)
    bayesian.py           execution-aware Bayesian expected-edge quoter
    rl_env.py / rl_agent.py  gym env + Double-DQN agent
  backtest/
    execution.py   queue-replay & closed-loop execution models
    engine.py      event-driven backtester + P&L decomposition
    metrics.py     performance & objective metrics
  report.py        one pipeline used by CLI, dashboard and tests
  plotting.py      paper figures
scripts/   run_pipeline · download_binance · make_dashboard · make_artifact · serve_dashboard
app/       FastAPI backend (app/backend) + React/Vite frontend (app/frontend)
dashboard/ streamlit_app.py
tests/     pytest suite (queue mechanics, survival consistency, P&L identities, …)
paper/     research_paper.md
```

## What is predicted (not "next candle up/down")

- **P(fill ≤ 100 ms / 500 ms / 1 s)** given queue position and market state
  (competing-risks survival model + closed-form Bayesian queue model).
- **Expected markout after execution** — the adverse-selection cost conditional
  on a fill, by queue position and distance.
- **P(cancellation / requote before execution)** — the competing risk.
- **Expected edge conditional on queue position**, which drives quoting.
- **Optimal size** given inventory and liquidity (Bayesian expected-utility and
  RL action selection).

## Objective

```
max  E[ spread revenue − adverse selection − fees − inventory cost ]
```

Each term is reported per strategy in `summarize_result` / the comparison table
and decomposed in the dashboards.

## Reproducibility

Every run is seeded (`config/default.yaml: seed`). `run_full_pipeline` is the
single entry point shared by the CLI, dashboard and tests, so the paper, the
dashboard and `pytest` all report identical numbers. Synthetic data and the RL
agent are cached under `data/cache/` and `results/`.

## Tests

```bash
pytest -q
```

## Notes & limitations

- The synthetic market is a reduced-form **aggregate** L2 book (size per level),
  not an agent-level one; queue position of *our* orders is tracked exactly.
- Real Binance public data is top-of-book depth; depth-aware features degrade
  gracefully to L1.
- The simulator is pure Python/NumPy+Numba. The brief suggests a C++ core for
  speed; the event loop is isolated in `qamm/lob/simulator.py` and could be
  swapped for a C++/pybind implementation without touching the models.
```
