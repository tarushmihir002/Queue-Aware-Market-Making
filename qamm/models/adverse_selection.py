"""
Adverse selection = the expected loss on a passive fill because the price
tends to move *against* us right after we get filled.

Measured by the **markout** at horizon h:
    markout_h = side · ( mid(t_fill + h) − p_fill )
A positive markout means the price went in our favour (we bought and it rose).
Adverse selection cost at horizon h is  −E[markout_h | fill].

We estimate E[markout_h | state, queue position] with a gradient-boosted
regressor on the probe dataset (fills only), and also report a bucketed
table by queue ratio and distance, which is the key diagnostic linking queue
position to the toxicity of fills.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .survival import CAUSE_FILL, ProbeDataset


class MarkoutModel:
    """E[markout_h | x] for a set of horizons. One model per horizon."""

    def __init__(self, horizons=(0.1, 0.5, 1.0, 5.0), kind: str = "gbr", seed: int = 0,
                 max_iter: int = 200, learning_rate: float = 0.05, max_depth: int = 4):
        self.horizons = list(horizons)
        self.kind = kind
        self.seed = seed
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth)
        self.models: Dict[float, object] = {}
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.r2_: Dict[float, float] = {}

    def _make(self):
        if self.kind == "ridge":
            return Ridge(alpha=1.0)
        return HistGradientBoostingRegressor(random_state=self.seed, l2_regularization=1.0, **self.params)

    def fit(self, ds: ProbeDataset, test_frac: float = 0.25) -> "MarkoutModel":
        m = ds.cause == CAUSE_FILL
        X = ds.X[m]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        n = len(Xn)
        perm = rng.permutation(n)
        n_te = int(test_frac * n)
        te, trn = perm[:n_te], perm[n_te:]
        for j, h in enumerate(ds.horizons):
            y = ds.markout[m, j]
            ok = np.isfinite(y)
            mdl = self._make()
            mdl.fit(Xn[trn][ok[trn]], y[trn][ok[trn]])
            self.models[float(h)] = mdl
            if n_te:
                yt = y[te][ok[te]]
                pr = mdl.predict(Xn[te][ok[te]])
                ss = np.sum((yt - yt.mean()) ** 2)
                self.r2_[float(h)] = float(1 - np.sum((yt - pr) ** 2) / ss) if ss > 0 else float("nan")
        self.horizons = [float(h) for h in ds.horizons]
        return self

    def predict(self, X: np.ndarray, horizon: float) -> np.ndarray:
        Xn = (np.atleast_2d(X) - self.mu) / self.sd
        return self.models[float(horizon)].predict(Xn)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "MarkoutModel":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def markout_table(ds: ProbeDataset, tick: float, horizon_idx: int = 2,
                  queue_edges=(0.0, 0.25, 0.5, 1.0, 10.0)) -> List[dict]:
    """Mean markout (in ticks) and fill rate by queue-ratio bucket and distance."""
    rows = []
    qe = np.asarray(queue_edges)
    for d in np.unique(ds.dist):
        for a, b in zip(qe[:-1], qe[1:]):
            m = (ds.dist == d) & (ds.queue_ratio >= a) & (ds.queue_ratio < b)
            if m.sum() < 20:
                continue
            f = m & (ds.cause == CAUSE_FILL)
            mo = ds.markout[f, horizon_idx] / tick
            rows.append({
                "dist_ticks": int(d), "queue_lo": float(a), "queue_hi": float(b), "n": int(m.sum()),
                "fill_rate": float(f.sum() / m.sum()),
                "p_fill_100ms": float(np.mean(ds.filled_within(0.1)[m])),
                "p_fill_500ms": float(np.mean(ds.filled_within(0.5)[m])),
                "p_fill_1s": float(np.mean(ds.filled_within(1.0)[m])),
                "p_requote": float(np.mean(ds.cause[m] == 2)),
                "median_fill_time": float(np.median(ds.time[f])) if f.any() else float("nan"),
                "markout_ticks": float(np.nanmean(mo)) if f.any() else float("nan"),
                "markout_se": float(np.nanstd(mo) / np.sqrt(max(f.sum(), 1))) if f.any() else float("nan"),
            })
    return rows
