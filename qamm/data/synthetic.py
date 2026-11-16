"""Convenience wrapper: generate (and cache) synthetic market data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .schema import MarketData


def generate_synthetic(cfg: dict, seed: int = 0, duration_s: Optional[float] = None,
                       cache_dir: str | Path | None = None, force: bool = False,
                       progress: bool = False) -> MarketData:
    from ..lob.simulator import HawkesLOBSimulator  # lazy: avoids data <-> lob import cycle

    cache = None
    if cache_dir is not None:
        d = duration_s or cfg["synthetic"]["duration_s"]
        cache = Path(cache_dir) / f"synthetic_{cfg['instrument']['symbol']}_seed{seed}_{int(d)}s.npz"
        if cache.exists() and not force:
            return MarketData.load(cache)
    sim = HawkesLOBSimulator(cfg, seed=seed)
    md = sim.run(T=duration_s, progress=progress)
    if cache is not None:
        md.save(cache)
    return md
