"""
Service layer: wraps the qamm pipeline so the API can run backtests on demand
and serve cached results. Runs are executed in a thread pool (the pipeline is
CPU-bound NumPy/Torch) and the most recent result per scenario is cached in
memory and mirrored to results/app_cache/ so a restart comes up instantly.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from qamm.report import bundle_to_json, run_full_pipeline
from qamm.strategy import DQNAgent
from qamm.utils import load_config

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "results" / "app_cache"
DATA_CACHE = ROOT / "data" / "cache"
PRECOMPUTED = ROOT / "results" / "run" / "dashboard_data.json"
AGENT_PATH = ROOT / "results" / "agent.pt"


# parameter ranges the frontend is allowed to set (defensive clamping)
BOUNDS = {
    "gamma": (0.005, 2.0), "maker_bps": (0.0, 5.0), "taker_bps": (0.0, 10.0),
    "quote_size": (0.1, 20.0), "momentum": (0.0, 0.8), "toxicity": (0.0, 0.8),
    "train_s": (300, 7200), "test_s": (300, 7200),
}


def _clamp(name: str, v: float) -> float:
    lo, hi = BOUNDS[name]
    return float(min(max(v, lo), hi))


class PipelineService:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._last_error: Optional[str] = None
        self._started_at: Optional[float] = None
        self._progress = "idle"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- status for the UI spinner
    def status(self) -> dict:
        return {"running": self._running, "progress": self._progress,
                "elapsed_s": (time.time() - self._started_at) if (self._running and self._started_at) else 0.0,
                "last_error": self._last_error,
                "has_rl_agent": AGENT_PATH.exists()}

    # ---- default / cached result shown on first load
    def default_result(self) -> Optional[dict]:
        latest = self._newest_cache()
        if latest is not None:
            return latest
        if PRECOMPUTED.exists():
            try:
                return json.loads(PRECOMPUTED.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def _newest_cache(self) -> Optional[dict]:
        files = sorted(CACHE_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

    # ---- the actual run (blocking; called inside a worker thread)
    def run(self, params: dict) -> dict:
        # Pydantic model_dump() includes unset fields as None; drop them so `.get(k, default)` works.
        params = {k: v for k, v in params.items() if v is not None}
        with self._lock:
            if self._running:
                raise RuntimeError("a backtest is already running")
            self._running = True
            self._started_at = time.time()
            self._last_error = None
            self._progress = "starting"
        try:
            cfg = self._build_config(params)
            include_rl = bool(params.get("include_rl", False))
            rl_agent = None
            if include_rl and AGENT_PATH.exists():
                self._progress = "loading RL agent"
                rl_agent = DQNAgent.load(AGENT_PATH, seed=0)
            self._progress = "generating data + fitting models"
            bundle = run_full_pipeline(
                cfg, train_seed=int(params.get("train_seed", 1)), test_seed=int(params.get("test_seed", 2)),
                train_duration_s=_clamp("train_s", params.get("train_s", 900)),
                test_duration_s=_clamp("test_s", params.get("test_s", 900)),
                fit_hawkes=bool(params.get("fit_hawkes", False)), rl_agent=rl_agent,
                cache_dir=str(DATA_CACHE), verbose=False,
            )
            self._progress = "serialising"
            js = bundle_to_json(bundle)
            js["params_used"] = {"gamma": cfg["strategy"]["gamma"], "maker_bps": cfg["fees"]["maker_bps"],
                                 "taker_bps": cfg["fees"]["taker_bps"], "quote_size": cfg["strategy"]["quote_size"],
                                 "momentum": cfg["hawkes"]["alpha"][0][0], "toxicity": cfg["hawkes"]["alpha"][4][1]}
            js["_meta"] = {"params": params, "generated_at": time.time()}
            self._save(js)
            return js
        except Exception as e:  # surface to the UI
            self._last_error = f"{type(e).__name__}: {e}"
            raise
        finally:
            with self._lock:
                self._running = False
                self._progress = "idle"

    def _save(self, js: dict) -> None:
        name = f"run_{int(time.time())}.json"
        (CACHE_DIR / name).write_text(json.dumps(js, default=float), encoding="utf-8")
        # keep only the 12 most recent
        files = sorted(CACHE_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[12:]:
            old.unlink(missing_ok=True)

    def _build_config(self, params: dict) -> dict:
        cfg = load_config()
        cfg["strategy"]["gamma"] = _clamp("gamma", params.get("gamma", cfg["strategy"]["gamma"]))
        cfg["strategy"]["quote_size"] = _clamp("quote_size", params.get("quote_size", cfg["strategy"]["quote_size"]))
        cfg["fees"]["maker_bps"] = _clamp("maker_bps", params.get("maker_bps", cfg["fees"]["maker_bps"]))
        cfg["fees"]["taker_bps"] = _clamp("taker_bps", params.get("taker_bps", cfg["fees"]["taker_bps"]))
        # market regime: reach into the Hawkes alpha matrix
        a = np.array(cfg["hawkes"]["alpha"], dtype=float)
        if "momentum" in params:
            a[0, 0] = a[1, 1] = _clamp("momentum", params["momentum"])
        if "toxicity" in params:
            a[4, 1] = a[5, 0] = _clamp("toxicity", params["toxicity"])
        cfg["hawkes"]["alpha"] = a.tolist()
        # smaller probe set for snappier interactive runs unless overridden
        cfg["fill_model"]["n_probe_orders"] = int(params.get("n_probes", 8000))
        return cfg


SERVICE = PipelineService()
