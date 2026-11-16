import numpy as np

from qamm.backtest import BacktestEngine, ReplayExecution, summarize_result
from qamm.models import (BayesianQueueModel, CompetingRiskModel, MarkoutModel, generate_probe_dataset)
from qamm.strategy import AvellanedaStoikov, BayesianQuoter, JoinTouchStrategy, calibrate_kappa


def _fit(market, features, cfg):
    ds = generate_probe_dataset(market, n_probes=2500, max_wait_s=5.0, order_size=1.0, seed=0,
                                features=features, lot=0.01)
    cr = CompetingRiskModel(epochs=8).fit(ds.X, ds.time, ds.cause)
    mm = MarkoutModel(horizons=ds.horizons).fit(ds)
    qm = BayesianQueueModel(m0=5.0).fit_from_data(market)
    return ds, cr, mm, qm


def test_as_quotes_are_passive_and_skewed(cfg, market, features):
    strat = AvellanedaStoikov(cfg)
    eng = BacktestEngine(cfg, ReplayExecution(market, start_ts=1.0), strat, features=features, max_duration_s=120)
    r = eng.run()
    assert r.n_bid_quotes > 0 and r.n_ask_quotes > 0


def test_kappa_positive(market, features, cfg):
    ds, *_ = _fit(market, features, cfg)
    k = calibrate_kappa(ds)
    assert k > 0


def test_bayesian_beats_baseline_objective(cfg, market, features):
    """On the toxic synthetic market the Bayesian quoter should not be worse than
    naive Avellaneda–Stoikov on the net objective (the project's whole point)."""
    ds, cr, mm, qm = _fit(market, features, cfg)
    res = {}
    for strat in [AvellanedaStoikov(cfg, kappa=calibrate_kappa(ds)), BayesianQuoter(cfg, qm, mm, cr)]:
        eng = BacktestEngine(cfg, ReplayExecution(market, start_ts=1.0), strat, features=features, max_duration_s=250)
        res[strat.name] = summarize_result(eng.run(), market.tick_size)
    assert res["bayesian_queue"]["objective"] >= res["avellaneda_stoikov"]["objective"] - 1e-6
    # its edge comes from taking less adverse selection per unit traded
    bq, as_ = res["bayesian_queue"], res["avellaneda_stoikov"]
    adv_per_fill_bq = bq["adverse_selection_1s"] / max(bq["n_fills"], 1)
    adv_per_fill_as = as_["adverse_selection_1s"] / max(as_["n_fills"], 1)
    assert adv_per_fill_bq <= adv_per_fill_as + 1e-6
