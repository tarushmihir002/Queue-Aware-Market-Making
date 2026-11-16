import numpy as np

from qamm.models import (BayesianQueueModel, CompetingRiskModel, aalen_johansen, generate_probe_dataset,
                         CAUSE_FILL, CAUSE_REQUOTE)


def test_probe_dataset_shapes(market, features):
    ds = generate_probe_dataset(market, n_probes=1500, max_wait_s=5.0, order_size=1.0, seed=0,
                                features=features, lot=0.01)
    assert ds.n == 1500
    assert ds.X.shape[1] == len(ds.feature_names)
    assert set(np.unique(ds.cause)).issubset({0, 1, 2})
    # closer to the touch / shorter queue should fill more often than deep-in-queue
    at_touch = ds.dist == 0
    front_m = at_touch & (ds.queue_ratio < 0.3)
    back_m = at_touch & (ds.queue_ratio >= 0.7)
    assert front_m.sum() > 20 and back_m.sum() > 20          # both buckets populated
    front = ds.filled_within(1.0)[front_m].mean()
    back = ds.filled_within(1.0)[back_m].mean()
    assert front >= back - 0.05


def test_aalen_johansen_monotone(market, features):
    ds = generate_probe_dataset(market, n_probes=2000, max_wait_s=5.0, order_size=1.0, seed=1,
                                features=features, lot=0.01)
    g = np.linspace(0.1, 5, 30)
    aj = aalen_johansen(ds.time, ds.cause, g)
    assert np.all(np.diff(aj["cif_fill"]) >= -1e-9)      # CIFs non-decreasing
    assert np.all(np.diff(aj["cif_requote"]) >= -1e-9)
    assert np.all(np.diff(aj["surv"]) <= 1e-9)           # survival non-increasing
    assert aj["cif_fill"][-1] + aj["cif_requote"][-1] <= 1.0 + 1e-6


def test_competing_risk_model_cif_consistency(market, features):
    ds = generate_probe_dataset(market, n_probes=3000, max_wait_s=5.0, order_size=1.0, seed=2,
                                features=features, lot=0.01)
    cr = CompetingRiskModel(epochs=8).fit(ds.X, ds.time, ds.cause)
    cf, crq, sv = cr.cif(ds.X[:200])
    tot = cf[:, -1] + crq[:, -1] + sv[:, -1]
    assert np.allclose(tot, 1.0, atol=1e-5)              # probabilities partition
    assert np.all(np.diff(cf, axis=1) >= -1e-6)          # monotone CIF
    p1 = cr.prob_fill_within(ds.X[:200], 1.0)
    assert np.all((p1 >= -1e-6) & (p1 <= 1 + 1e-6))


def test_bayesian_queue_monotone_in_q():
    qm = BayesianQueueModel(a0=4.0, b0=2.0, m0=5.0)
    taus = np.array([0.1, 0.5, 1.0, 2.0])
    # fill prob decreases as queue ahead grows, increases with tau
    p_small = qm.prob_fill_curve(1, q=5.0, taus=taus)
    p_big = qm.prob_fill_curve(1, q=50.0, taus=taus)
    assert np.all(p_small >= p_big - 1e-9)
    assert np.all(np.diff(p_small) >= -1e-9)
    assert qm.prob_fill_within(1, 0.0, 1.0) == 1.0       # nothing ahead -> certain fill


def test_bayesian_update_moves_rate():
    qm = BayesianQueueModel(a0=2.0, b0=1.0, m0=5.0)
    r0 = qm.rate_mean(1)
    qm.update(1, n_events=100, exposure_s=10.0, sizes=np.full(100, 5.0))
    assert qm.rate_mean(1) > r0                          # observing many events raises the posterior rate
