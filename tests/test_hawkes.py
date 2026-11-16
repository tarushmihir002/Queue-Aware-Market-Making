import numpy as np

from qamm.hawkes import HawkesParams, MultivariateHawkes, branching_ratio, fit_hawkes_mle


def test_branching_ratio_stationary():
    cfg_like = {"mu": [1.0], "alpha": [[0.5]], "beta": 2.0}
    p = HawkesParams.from_config(cfg_like)
    assert abs(p.branching_ratio() - 0.25) < 1e-9


def test_non_stationary_rejected():
    p = HawkesParams(np.array([1.0]), np.array([[3.0]]), np.array([[2.0]]))
    try:
        MultivariateHawkes(p)
        assert False, "should reject non-stationary params"
    except ValueError:
        pass


def test_simulate_rate_matches_theory():
    # univariate: stationary mean rate = mu / (1 - n)
    mu, alpha, beta = 1.0, 0.5, 2.0
    p = HawkesParams(np.array([mu]), np.array([[alpha]]), np.array([[beta]]))
    h = MultivariateHawkes(p)
    t, k = h.simulate(4000.0, seed=1)
    rate = len(t) / 4000.0
    theory = mu / (1 - alpha / beta)   # = 1 / (1-0.25) = 1.333
    assert abs(rate - theory) / theory < 0.1


def test_mle_recovers_parameters():
    p = HawkesParams(np.array([0.8, 0.6]), np.array([[0.4, 0.1], [0.05, 0.3]]), np.full((2, 2), 2.0))
    h = MultivariateHawkes(p)
    t, k = h.simulate(6000.0, seed=3)
    est = fit_hawkes_mle(t, k, D=2, T=6000.0, beta_init=2.0, shared_beta=True)
    # mu within 25%, branching ratio in right ballpark
    assert np.all(np.abs(est.mu - p.mu) / p.mu < 0.3)
    assert abs(est.branching_ratio() - p.branching_ratio()) < 0.15
