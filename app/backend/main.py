"""
FastAPI server for the queue-aware market-making dashboard.

Endpoints
---------
  GET  /api/health              liveness
  GET  /api/config              parameter bounds + defaults for the control panel
  GET  /api/status              whether a backtest is running (for the spinner)
  GET  /api/results             the most recent / precomputed result bundle
  POST /api/run                 run a backtest with the posted parameters

It also serves the built React app from app/frontend/dist at "/", so a single
`uvicorn app.backend.main:app` hosts the whole dashboard locally.

Run:
  uvicorn app.backend.main:app --port 8000           # serves API + built frontend
  # dev: run the Vite dev server separately (npm run dev), it proxies /api here
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.backend.service import BOUNDS, SERVICE
from qamm.utils import load_config

app = FastAPI(title="Queue-Aware Market Making Dashboard", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_pool = ThreadPoolExecutor(max_workers=1)
FRONTEND_DIST = ROOT / "app" / "frontend" / "dist"


class RunParams(BaseModel):
    gamma: float | None = None
    maker_bps: float | None = None
    taker_bps: float | None = None
    quote_size: float | None = None
    momentum: float | None = None
    toxicity: float | None = None
    train_s: float | None = None
    test_s: float | None = None
    train_seed: int | None = None
    test_seed: int | None = None
    n_probes: int | None = None
    include_rl: bool = False
    fit_hawkes: bool = False


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def config():
    cfg = load_config()
    return {
        "bounds": BOUNDS,
        "defaults": {
            "gamma": cfg["strategy"]["gamma"], "maker_bps": cfg["fees"]["maker_bps"],
            "taker_bps": cfg["fees"]["taker_bps"], "quote_size": cfg["strategy"]["quote_size"],
            "momentum": cfg["hawkes"]["alpha"][0][0], "toxicity": cfg["hawkes"]["alpha"][4][1],
            "train_s": 900, "test_s": 900, "n_probes": 8000, "include_rl": False,
        },
        "symbol": cfg["instrument"]["symbol"], "tick_size": cfg["instrument"]["tick_size"],
    }


@app.get("/api/status")
def status():
    return SERVICE.status()


@app.get("/api/results")
def results():
    r = SERVICE.default_result()
    if r is None:
        return JSONResponse({"empty": True, "message": "No results yet — run a backtest."})
    return r


@app.post("/api/run")
def run(params: RunParams):
    if SERVICE.status()["running"]:
        raise HTTPException(status_code=409, detail="A backtest is already running.")
    fut = _pool.submit(SERVICE.run, params.model_dump())
    try:
        return fut.result()  # block until done; the frontend shows a spinner
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# ---- serve the built React frontend (if present) ----------------------------
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        # client-side routing fallback
        f = FRONTEND_DIST / full_path
        if f.is_file():
            return FileResponse(f)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    def no_frontend():
        return JSONResponse({
            "message": "Frontend not built. Run `npm install && npm run build` in app/frontend, "
                       "or use the Vite dev server (`npm run dev`).",
            "api": ["/api/health", "/api/config", "/api/results", "/api/run", "/api/status"],
        })
