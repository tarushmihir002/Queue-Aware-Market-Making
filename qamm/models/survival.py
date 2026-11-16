"""
Fill-probability models for passive limit orders.

We treat the lifetime of a resting order as a *competing-risks* problem:

    cause 1  FILL     — the queue ahead is consumed and we execute
    cause 2  REQUOTE  — the best price on our side moves past us; a market
                        maker would cancel and re-quote (this is the
                        "cancellation before execution" risk)
    censored         — still resting at the maximum wait time

Three estimators, from least to most structured:

1. `aalen_johansen`            non-parametric cumulative-incidence functions
                               (population or bucket level; model-free check)
2. `CompetingRiskModel`        discrete-time competing-risk neural model:
                               P(cause c in bin k | survived to k, features).
                               Gives P(fill ≤ τ | state, queue position) for
                               τ = 100 ms, 500 ms, 1 s and P(requote ≤ τ).
3. `BayesianQueueModel`        closed-form Bayesian queue-depletion model. The
                               queue ahead (q) is consumed by a compound-Poisson
                               stream (rate λ, exponential sizes mean m).
                               With a conjugate Gamma posterior on λ,
                                 P(fill ≤ τ | q) = Σ_n NB(n; a, b/(b+τ)) · Q(n, q/m)
                               where Q is the regularised upper incomplete gamma
                               function. λ is updated online from observed
                               trade flow, which is what makes the Bayesian
                               market maker adapt within the session.

`generate_probe_dataset` builds the labelled dataset by placing *virtual*
orders in recorded data and replaying their queue position forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.special import gammaincc, gammaln

from ..backtest.execution import (OUTCOME_FILL, OUTCOME_NONE, OUTCOME_REQUOTE, VirtualOrder,
                                  replay_order_step)
from ..data.schema import MarketData
from .features import FEATURE_NAMES, ORDER_FEATURE_NAMES, compute_features, order_features

CAUSE_NONE, CAUSE_FILL, CAUSE_REQUOTE = OUTCOME_NONE, OUTCOME_FILL, OUTCOME_REQUOTE


# ---------------------------------------------------------------------------
@dataclass
class ProbeDataset:
    X: np.ndarray              # (n, F) market + order features at placement
    time: np.ndarray           # (n,) time to event / censoring (s)
    cause: np.ndarray          # (n,) 0 censored, 1 fill, 2 requote
    markout: np.ndarray        # (n, H) side·(mid(t_fill+h) − price); NaN if not filled
    horizons: np.ndarray       # (H,)
    side: np.ndarray           # (n,)
    dist: np.ndarray           # (n,) ticks from best
    queue_ratio: np.ndarray    # (n,)
    place_idx: np.ndarray      # (n,) book row of placement
    feature_names: List[str]

    @property
    def n(self) -> int:
        return len(self.time)

    def filled_within(self, tau: float) -> np.ndarray:
        return (self.cause == CAUSE_FILL) & (self.time <= tau)

    def save(self, path):
        np.savez_compressed(path, X=self.X, time=self.time, cause=self.cause, markout=self.markout,
                            horizons=self.horizons, side=self.side, dist=self.dist, queue_ratio=self.queue_ratio,
                            place_idx=self.place_idx, feature_names=np.array(self.feature_names))

    @classmethod
    def load(cls, path) -> "ProbeDataset":
        z = np.load(path, allow_pickle=False)
        return cls(z["X"], z["time"], z["cause"], z["markout"], z["horizons"], z["side"], z["dist"],
                   z["queue_ratio"], z["place_idx"], list(z["feature_names"]))


def generate_probe_dataset(md: MarketData, n_probes: int = 20000, max_wait_s: float = 10.0,
                           order_size: float = 0.01, horizons=(0.1, 0.5, 1.0, 5.0),
                           dist_choices=(0, 0, 0, 1, 2), queue_frac_choices=(1.0, 1.0, 0.5, 0.25, 0.0),
                           seed: int = 0, features: Optional[np.ndarray] = None,
                           lot: float = 0.001) -> ProbeDataset:
    """Place virtual orders at random times and replay their queue position.

    dist_choices        distance (ticks) from the best on our side, 0 = join the touch
    queue_frac_choices  fraction of the resting level volume placed *ahead* of us
                        (1.0 = join the back, 0.0 = front of queue). Varying it is
                        what lets the model learn the queue-position effect.
    """
    rng = np.random.default_rng(seed)
    b, tr = md.book, md.trades
    tick = md.tick_size
    F = features if features is not None else compute_features(md)
    horizons = np.asarray(horizons, dtype=float)
    trade_ptr = np.searchsorted(tr.ts, b.ts, side="right")
    t_end = b.ts[-1] - max_wait_s - horizons.max() - 1.0
    cand = np.where(b.ts < t_end)[0]
    cand = cand[cand > 1]
    idx = rng.choice(cand, size=n_probes, replace=True)

    n = len(idx)
    X = np.zeros((n, len(FEATURE_NAMES) + len(ORDER_FEATURE_NAMES)), dtype=np.float32)
    time = np.zeros(n); cause = np.zeros(n, dtype=np.int8)
    markout = np.full((n, len(horizons)), np.nan)
    side_arr = np.zeros(n, dtype=np.int8); dist_arr = np.zeros(n, dtype=np.int8); qr = np.zeros(n)
    mid = b.mid

    for k, i0 in enumerate(idx):
        side = 1 if rng.random() < 0.5 else -1
        d = int(rng.choice(dist_choices))
        qf = float(rng.choice(queue_frac_choices))
        best = b.bid_px[i0, 0] if side > 0 else b.ask_px[i0, 0]
        price = round(best - side * d * tick, 10)
        px_row = b.bid_px[i0] if side > 0 else b.ask_px[i0]
        sz_row = b.bid_sz[i0] if side > 0 else b.ask_sz[i0]
        m = np.abs(px_row - price) < 1e-9
        level = float(sz_row[m].sum()) if m.any() else 0.0
        q = level * qf
        top_same = b.bid_sz[i0, 0] if side > 0 else b.ask_sz[i0, 0]
        X[k, :len(FEATURE_NAMES)] = F[i0]
        X[k, len(FEATURE_NAMES):] = order_features(side, d, q, top_same, F[i0, 1], lot)
        side_arr[k], dist_arr[k], qr[k] = side, d, q / max(top_same, 1e-9)

        o = VirtualOrder(0, side, price, order_size, q, b.ts[i0])
        t0 = b.ts[i0]
        i = i0
        out = CAUSE_NONE
        fills: List[Tuple[float, float]] = []
        while i + 1 < b.n and b.ts[i + 1] - t0 <= max_wait_s:
            i += 1
            out = replay_order_step(o, md, i, trade_ptr[i - 1], trade_ptr[i], fills)
            if out != CAUSE_NONE:
                break
        if out == CAUSE_FILL:
            tf = fills[-1][0]
            time[k] = max(tf - t0, 1e-6); cause[k] = CAUSE_FILL
            jj = np.searchsorted(b.ts, tf + horizons, side="right") - 1
            markout[k] = side * (mid[jj] - price)
        elif out == CAUSE_REQUOTE:
            time[k] = max(b.ts[i] - t0, 1e-6); cause[k] = CAUSE_REQUOTE
        else:
            time[k] = max_wait_s; cause[k] = CAUSE_NONE
    return ProbeDataset(X, time, cause, markout, horizons, side_arr, dist_arr, qr, idx,
                        FEATURE_NAMES + ORDER_FEATURE_NAMES)


# ---------------------------------------------------------------------------
def aalen_johansen(time: np.ndarray, cause: np.ndarray, grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Non-parametric cumulative incidence F_c(t) for each cause on `grid`.

    AJ estimator: F_c(t) = Σ_{t_i ≤ t} S(t_i−) · d_c(t_i)/n(t_i),  S = Kaplan–Meier
    of the event-free survival."""
    order = np.argsort(time)
    t, c = time[order], cause[order]
    ut = np.unique(t[c > 0])
    n_at_risk = len(t)
    S = 1.0
    F1 = F2 = 0.0
    cif1, cif2, surv = [], [], []
    ptr = 0
    g_ptr = 0
    out1 = np.zeros(len(grid)); out2 = np.zeros(len(grid)); outS = np.ones(len(grid))
    for tv in ut:
        # fill grid points before tv
        while g_ptr < len(grid) and grid[g_ptr] < tv:
            out1[g_ptr], out2[g_ptr], outS[g_ptr] = F1, F2, S
            g_ptr += 1
        # risk set
        while ptr < len(t) and t[ptr] < tv:
            ptr += 1
        n_at_risk = len(t) - ptr
        mask = t == tv
        d1 = np.sum(mask & (c == CAUSE_FILL)); d2 = np.sum(mask & (c == CAUSE_REQUOTE))
        if n_at_risk > 0:
            F1 += S * d1 / n_at_risk
            F2 += S * d2 / n_at_risk
            S *= 1 - (d1 + d2) / n_at_risk
    while g_ptr < len(grid):
        out1[g_ptr], out2[g_ptr], outS[g_ptr] = F1, F2, S
        g_ptr += 1
    return {"grid": grid, "cif_fill": out1, "cif_requote": out2, "surv": outS}


# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


class CompetingRiskModel:
    """Discrete-time competing-risk model.

    Time is binned with edges `bins`. For each order and each bin it survived
    into, the model predicts a 3-way softmax: {survive, fill, requote}. The
    cumulative incidence of fill by τ is
        CIF_fill(τ|x) = Σ_{k: e_k ≤ τ} S_{k-1}(x) · h_fill,k(x),
        S_k = Π_{l ≤ k} (1 − h_fill,l − h_req,l).
    """

    def __init__(self, bins=(0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0), hidden: int = 64,
                 epochs: int = 30, lr: float = 2e-3, batch: int = 2048, seed: int = 0,
                 weight_decay: float = 1e-5):
        self.bins = np.asarray(bins, dtype=float)
        self.K = len(self.bins) - 1
        self.hidden, self.epochs, self.lr, self.batch, self.seed, self.wd = hidden, epochs, lr, batch, seed, weight_decay
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.net: Optional[_MLP] = None
        self.history: List[float] = []

    # ---- person-period expansion
    def _expand(self, X, time, cause):
        rows, bins_, y = [], [], []
        kb = np.clip(np.searchsorted(self.bins, time, side="right") - 1, 0, self.K - 1)
        for i in range(len(time)):
            for k in range(kb[i] + 1):
                rows.append(i); bins_.append(k)
                if k == kb[i] and cause[i] != CAUSE_NONE:
                    y.append(int(cause[i]))
                else:
                    y.append(0)
        return np.asarray(rows), np.asarray(bins_), np.asarray(y)

    def _design(self, Xn: np.ndarray, k: np.ndarray) -> torch.Tensor:
        oh = np.zeros((len(k), self.K), dtype=np.float32)
        oh[np.arange(len(k)), k] = 1.0
        return torch.from_numpy(np.hstack([Xn, oh]).astype(np.float32))

    def fit(self, X: np.ndarray, time: np.ndarray, cause: np.ndarray, verbose: bool = False) -> "CompetingRiskModel":
        torch.manual_seed(self.seed)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xn = ((X - self.mu) / self.sd).astype(np.float32)
        rows, kk, y = self._expand(Xn, time, cause)
        D = self._design(Xn[rows], kk)
        Y = torch.from_numpy(y.astype(np.int64))
        self.net = _MLP(D.shape[1], 3, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        n = len(Y)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for s in range(0, n, self.batch):
                idx = perm[s:s + self.batch]
                logits = self.net(D[idx])
                loss = nn.functional.cross_entropy(logits, Y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            self.history.append(tot / n)
            if verbose and (ep % 5 == 0 or ep == self.epochs - 1):
                print(f"  epoch {ep:3d}  nll/row {tot / n:.4f}")
        return self

    @torch.no_grad()
    def hazards(self, X: np.ndarray) -> np.ndarray:
        """(n, K, 3) per-bin probabilities {survive, fill, requote}."""
        Xn = ((X - self.mu) / self.sd).astype(np.float32)
        n = len(Xn)
        out = np.zeros((n, self.K, 3), dtype=np.float32)
        self.net.eval()
        for k in range(self.K):
            D = self._design(Xn, np.full(n, k))
            out[:, k] = torch.softmax(self.net(D), dim=1).numpy()
        return out

    def cif(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cumulative incidence at each bin edge e_1..e_K. Returns (cif_fill, cif_requote, surv), each (n, K)."""
        h = self.hazards(X)
        S = np.ones(len(X))
        cf = np.zeros((len(X), self.K)); cr = np.zeros((len(X), self.K)); sv = np.zeros((len(X), self.K))
        for k in range(self.K):
            cf[:, k] = (cf[:, k - 1] if k else 0) + S * h[:, k, 1]
            cr[:, k] = (cr[:, k - 1] if k else 0) + S * h[:, k, 2]
            S = S * h[:, k, 0]
            sv[:, k] = S
        return cf, cr, sv

    def prob_fill_within(self, X: np.ndarray, tau: float) -> np.ndarray:
        """P(fill ≤ τ | x). τ is linearly interpolated between bin edges."""
        cf, _, _ = self.cif(X)
        return self._interp(cf, tau)

    def prob_requote_within(self, X: np.ndarray, tau: float) -> np.ndarray:
        _, cr, _ = self.cif(X)
        return self._interp(cr, tau)

    def _interp(self, c: np.ndarray, tau: float) -> np.ndarray:
        edges = self.bins[1:]
        full = np.hstack([np.zeros((len(c), 1)), c])
        e = np.hstack([[0.0], edges])
        tau = float(np.clip(tau, 0, edges[-1]))
        k = np.searchsorted(e, tau, side="right") - 1
        if k >= len(edges):
            return full[:, -1]
        w = (tau - e[k]) / (e[k + 1] - e[k])
        return full[:, k] * (1 - w) + full[:, k + 1] * w

    # ---- persistence
    def save(self, path):
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd, "bins": self.bins,
                    "hidden": self.hidden, "history": self.history}, path)

    @classmethod
    def load(cls, path, d_in: int) -> "CompetingRiskModel":
        ck = torch.load(path, weights_only=False)
        m = cls(bins=ck["bins"], hidden=ck["hidden"])
        m.mu, m.sd, m.history = ck["mu"], ck["sd"], ck["history"]
        m.net = _MLP(d_in + m.K, 3, m.hidden)
        m.net.load_state_dict(ck["state"])
        return m


# ---------------------------------------------------------------------------
class BayesianQueueModel:
    """Closed-form Bayesian fill probability for an order with volume q ahead.

    Consumption of the queue ahead at our price arrives as events of a Poisson
    process with rate λ (per second) and i.i.d. Exp(mean m) sizes:
        P(C(τ) ≥ q | λ) = Σ_n Pois(n; λτ) · P(Gamma(n, m) ≥ q)
                        = Σ_n Pois(n; λτ) · Q(n, q/m)
    with Q(n, z) = Γ(n, z)/Γ(n) the regularised upper incomplete gamma
    (Q(0, z) = 0 for z > 0).

    λ is unknown. Prior λ ~ Gamma(a0, b0) (shape, rate). Observing n events in
    an exposure of Δ seconds gives the posterior Gamma(a0 + n, b0 + Δ), and the
    posterior predictive for N(τ) is Negative-Binomial:
        P(N = n) = C(a+n−1, n) · (b/(b+τ))^a · (τ/(b+τ))^n.
    Hence  P(fill ≤ τ | q) = Σ_n NB(n; a, b/(b+τ)) · Q(n, q/m).

    `update(n_events, exposure, sizes)` performs the conjugate update with
    exponential forgetting so the rate tracks intra-day regime changes. The
    same object keeps one posterior per side (bid queue is eaten by sells).
    """

    def __init__(self, a0: float = 2.0, b0: float = 1.0, m0: float = 0.05, forget: float = 0.98,
                 n_max: int = 200):
        self.a = np.array([a0, a0], dtype=float)   # index 0: bid side (eaten by sell trades), 1: ask side
        self.b = np.array([b0, b0], dtype=float)
        self.m = np.array([m0, m0], dtype=float)
        self.m_n = np.array([5.0, 5.0])          # pseudo-count for the size mean
        self.forget = forget
        self.n_max = n_max
        self.n_grid = np.arange(n_max + 1)

    @staticmethod
    def _side_idx(side: int) -> int:
        return 0 if side > 0 else 1

    def update(self, side: int, n_events: int, exposure_s: float, sizes: Optional[np.ndarray] = None) -> None:
        s = self._side_idx(side)
        # forgetting shrinks the posterior towards the prior-like diffuse state
        self.a[s] = self.forget * self.a[s] + n_events
        self.b[s] = self.forget * self.b[s] + exposure_s
        if sizes is not None and len(sizes):
            w = self.forget * self.m_n[s]
            self.m[s] = (w * self.m[s] + sizes.sum()) / (w + len(sizes))
            self.m_n[s] = w + len(sizes)

    def rate_posterior(self, side: int) -> Tuple[float, float]:
        s = self._side_idx(side)
        return self.a[s], self.b[s]

    def rate_mean(self, side: int) -> float:
        s = self._side_idx(side)
        return self.a[s] / self.b[s]

    def sample_rate(self, side: int, rng: np.random.Generator) -> float:
        s = self._side_idx(side)
        return float(rng.gamma(self.a[s], 1.0 / self.b[s]))

    def prob_fill_within(self, side: int, q: float, tau: float, lam: Optional[float] = None) -> float:
        """Posterior-predictive (lam=None) or plug-in (lam given) fill probability."""
        s = self._side_idx(side)
        z = max(q, 0.0) / self.m[s]
        n = self.n_grid
        # P(Gamma(n, m) >= q) = Q(n, q/m); n=0 -> 0 unless q == 0
        Qn = gammaincc(np.maximum(n, 1e-12), z)
        Qn[0] = 1.0 if q <= 1e-12 else 0.0
        if lam is None:
            a, b = self.a[s], self.b[s]
            p = b / (b + tau)
            logpmf = gammaln(a + n) - gammaln(a) - gammaln(n + 1) + a * np.log(p) + n * np.log1p(-p)
            pmf = np.exp(logpmf)
        else:
            mu = lam * tau
            logpmf = n * np.log(max(mu, 1e-300)) - mu - gammaln(n + 1)
            pmf = np.exp(logpmf)
        return float(np.clip((pmf * Qn).sum(), 0.0, 1.0))

    def prob_fill_curve(self, side: int, q: float, taus: np.ndarray) -> np.ndarray:
        return np.array([self.prob_fill_within(side, q, t) for t in taus])

    def fit_from_data(self, md: MarketData, window_s: float = 60.0) -> "BayesianQueueModel":
        """Initialise from the empirical trade stream (rate & mean size per side)."""
        tr = md.trades
        T = md.book.ts[-1] - md.book.ts[0]
        for side, agg in ((1, -1), (-1, 1)):   # bids are eaten by sell aggressors
            m = tr.side == agg
            n = int(m.sum())
            self.update(side, n, max(T, 1e-6), tr.size[m])
        return self


# ---------------------------------------------------------------------------
def evaluate_fill_model(p_hat: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    """AUC, Brier, log-loss and a reliability table for binary fill-within-τ."""
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    p = np.clip(p_hat, 1e-6, 1 - 1e-6)
    out = {
        "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "base_rate": float(y.mean()),
        "brier_const": float(np.mean((y - y.mean()) ** 2)),
    }
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    k = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, n_bins - 1)
    rel = [(float(p[k == j].mean()), float(y[k == j].mean()), int((k == j).sum()))
           for j in range(n_bins) if (k == j).any()]
    out["reliability"] = rel   # (mean predicted, observed, count)
    return out
