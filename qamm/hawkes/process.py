"""
Multivariate Hawkes process with exponential kernels.

    λ_i(t) = μ_i + Σ_j Σ_{t_k^j < t} α_ij · exp(−β_ij (t − t_k^j))

* Simulation  : Ogata's modified thinning algorithm (exact).
* Estimation  : maximum likelihood with the O(N·D²) recursion for exponential
                kernels (no numerical integration needed).
* Diagnostics : branching ratio (spectral radius of α/β) – must be < 1 for
                stationarity; residual analysis via the random time change
                theorem (compensator increments should be Exp(1)).

The LOB simulator uses D = 6 event types (market buy/sell, limit buy/sell add,
cancel buy/sell). Nothing here is LOB-specific though.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize


@dataclass
class HawkesParams:
    mu: np.ndarray      # (D,)
    alpha: np.ndarray   # (D, D)  alpha[i, j] = effect of type j on type i
    beta: np.ndarray    # (D, D)

    @classmethod
    def from_config(cls, hcfg: dict) -> "HawkesParams":
        mu = np.asarray(hcfg["mu"], dtype=float)
        alpha = np.asarray(hcfg["alpha"], dtype=float)
        beta = hcfg["beta"]
        beta = np.full_like(alpha, float(beta)) if np.isscalar(beta) else np.asarray(beta, dtype=float)
        return cls(mu, alpha, beta)

    @property
    def dim(self) -> int:
        return len(self.mu)

    def branching_ratio(self) -> float:
        return branching_ratio(self.alpha, self.beta)

    def to_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "alpha": self.alpha.tolist(), "beta": self.beta.tolist()}


def branching_ratio(alpha: np.ndarray, beta: np.ndarray) -> float:
    """Spectral radius of the branching matrix α/β. < 1 ⇒ stationary."""
    G = alpha / beta
    return float(np.max(np.abs(np.linalg.eigvals(G))))


# ---------------------------------------------------------------------------
# Simulation — Ogata thinning
# ---------------------------------------------------------------------------
@njit(cache=True)
def _simulate(mu, alpha, beta, T, seed, max_events):
    np.random.seed(seed)
    D = mu.shape[0]
    # A[i, j] = Σ_{t_k^j < t} exp(-β_ij (t - t_k^j)) evaluated at current time
    A = np.zeros((D, D))
    times = np.empty(max_events)
    types = np.empty(max_events, dtype=np.int64)
    n = 0
    t = 0.0
    lam = mu.copy()
    while n < max_events:
        lam_bar = lam.sum()                # upper bound: intensity is non-increasing until next event
        w = np.random.exponential(1.0 / lam_bar)
        t_new = t + w
        if t_new > T:
            break
        # decay A to t_new
        for i in range(D):
            for j in range(D):
                A[i, j] *= np.exp(-beta[i, j] * w)
        for i in range(D):
            s = mu[i]
            for j in range(D):
                s += alpha[i, j] * A[i, j]
            lam[i] = s
        t = t_new
        u = np.random.random() * lam_bar
        if u <= lam.sum():
            # accept; pick type
            c = 0.0
            k = D - 1
            for i in range(D):
                c += lam[i]
                if u <= c:
                    k = i
                    break
            times[n] = t
            types[n] = k
            n += 1
            for i in range(D):
                A[i, k] += 1.0
                lam[i] += alpha[i, k]
    return times[:n], types[:n]


class MultivariateHawkes:
    def __init__(self, params: HawkesParams):
        self.p = params
        if params.branching_ratio() >= 1.0:
            raise ValueError(f"Hawkes process not stationary: branching ratio = {params.branching_ratio():.3f}")

    def simulate(self, T: float, seed: int = 0, max_events: int | None = None):
        """Return (times, types) on [0, T]."""
        if max_events is None:
            # expected count = μ T / (1 - n) with n = branching ratio; add generous head-room
            n_exp = self.p.mu.sum() * T / max(1e-6, 1.0 - self.p.branching_ratio())
            max_events = int(3 * n_exp + 10_000)
        t, k = _simulate(self.p.mu, self.p.alpha, self.p.beta, float(T), int(seed), int(max_events))
        return t, k

    def intensity_path(self, times: np.ndarray, types: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """λ_i(t) evaluated on `grid` (D, len(grid)) — for plotting / features."""
        D = self.p.dim
        out = np.zeros((D, len(grid)))
        for i in range(D):
            out[i] = self.p.mu[i]
            for j in range(D):
                tj = times[types == j]
                if len(tj) == 0:
                    continue
                # vectorised: for each grid point, sum over events before it
                idx = np.searchsorted(tj, grid, side="left")
                # recursion over grid to avoid O(N·G)
                a = 0.0
                prev_t = -np.inf
                ptr = 0
                vals = np.empty(len(grid))
                for g in range(len(grid)):
                    tg = grid[g]
                    # decay from prev_t
                    if np.isfinite(prev_t):
                        a *= np.exp(-self.p.beta[i, j] * (tg - prev_t))
                    while ptr < idx[g]:
                        a += np.exp(-self.p.beta[i, j] * (tg - tj[ptr]))
                        ptr += 1
                    vals[g] = a
                    prev_t = tg
                out[i] += self.p.alpha[i, j] * vals
        return out


# ---------------------------------------------------------------------------
# Estimation — exponential-kernel MLE
# ---------------------------------------------------------------------------
@njit(cache=True)
def _neg_loglik(mu, alpha, beta, times, types, T):
    D = mu.shape[0]
    A = np.zeros((D, D))
    ll = 0.0
    prev = 0.0
    n = times.shape[0]
    for k in range(n):
        t = times[k]
        m = types[k]
        dt = t - prev
        for i in range(D):
            for j in range(D):
                A[i, j] *= np.exp(-beta[i, j] * dt)
        lam = mu[m]
        for j in range(D):
            lam += alpha[m, j] * A[m, j]
        if lam <= 1e-12:
            return 1e18
        ll += np.log(lam)
        for i in range(D):
            A[i, m] += 1.0
        prev = t
    # compensator
    comp = mu.sum() * T
    for k in range(n):
        j = types[k]
        rem = T - times[k]
        for i in range(D):
            comp += alpha[i, j] / beta[i, j] * (1.0 - np.exp(-beta[i, j] * rem))
    return -(ll - comp)


def fit_hawkes_mle(times: np.ndarray, types: np.ndarray, D: int, T: float | None = None,
                   beta_init: float = 2.0, fit_beta: bool = True, shared_beta: bool = True,
                   maxiter: int = 300, verbose: bool = False) -> HawkesParams:
    """MLE for (μ, α, β) with exponential kernels.

    Parameters are optimised in log-space (positivity) with L-BFGS-B. With
    `shared_beta=True` a single decay β is estimated (robust, few parameters);
    otherwise a per-(i,j) β.
    """
    times = np.asarray(times, dtype=float)
    types = np.asarray(types, dtype=np.int64)
    if T is None:
        T = float(times[-1])
    counts = np.bincount(types, minlength=D).astype(float)
    mu0 = np.maximum(counts / T * 0.5, 1e-3)
    alpha0 = np.full((D, D), 0.1)
    nb = 1 if shared_beta else D * D

    def unpack(x):
        mu = np.exp(x[:D])
        alpha = np.exp(x[D:D + D * D]).reshape(D, D)
        if fit_beta:
            b = np.exp(x[D + D * D:])
            beta = np.full((D, D), b[0]) if shared_beta else b.reshape(D, D)
        else:
            beta = np.full((D, D), beta_init)
        return mu, alpha, beta

    x0 = np.concatenate([np.log(mu0), np.log(alpha0).ravel(),
                         np.log(np.full(nb, beta_init)) if fit_beta else np.array([])])

    def f(x):
        mu, alpha, beta = unpack(x)
        return _neg_loglik(mu, alpha, beta, times, types, T)

    res = minimize(f, x0, method="L-BFGS-B", options={"maxiter": maxiter, "disp": verbose})
    mu, alpha, beta = unpack(res.x)
    return HawkesParams(mu, alpha, beta)


def residuals(params: HawkesParams, times: np.ndarray, types: np.ndarray, kind: int) -> np.ndarray:
    """Random-time-change residuals for event type `kind`: Λ(t_k) − Λ(t_{k-1}) should be iid Exp(1)."""
    D = params.dim
    tk = times[types == kind]
    comp = np.zeros(len(tk))
    for n_, t in enumerate(tk):
        c = params.mu[kind] * t
        for j in range(D):
            tj = times[(types == j) & (times < t)]
            c += np.sum(params.alpha[kind, j] / params.beta[kind, j] * (1 - np.exp(-params.beta[kind, j] * (t - tj))))
        comp[n_] = c
    return np.diff(comp)
