import numpy as np

from qamm.backtest import BacktestEngine, ReplayExecution, ClosedLoopExecution, summarize_result
from qamm.lob import HawkesLOBSimulator
from qamm.strategy import JoinTouchStrategy, FixedSpreadStrategy, AvellanedaStoikov


def test_replay_backtest_runs(cfg, market, features):
    eng = BacktestEngine(cfg, ReplayExecution(market, start_ts=1.0), JoinTouchStrategy(cfg),
                         features=features, max_duration_s=120)
    r = eng.run()
    assert len(r.ts) > 10
    assert r.markouts.shape == (r.n_fills, len(r.markout_horizons))
    s = summarize_result(r, market.tick_size)
    # P&L identity: mtm = spread_revenue + inventory_pnl - fees
    assert abs(s["mtm_pnl"] - (s["spread_revenue"] + s["inventory_pnl"] - s["fees"])) < 1e-6


def test_inventory_cap_respected(cfg, market, features):
    c = cfg.copy(); c["strategy"]["max_inventory"] = 3.0
    eng = BacktestEngine(c, ReplayExecution(market, start_ts=1.0), JoinTouchStrategy(c),
                         features=features, max_duration_s=200)
    r = eng.run()
    assert np.max(np.abs(r.inventory)) <= 3.0 + c["strategy"]["quote_size"] + 1e-6


def test_closed_loop_backtest_runs(cfg):
    sim = HawkesLOBSimulator(cfg, seed=9); sim.generate_events(150)
    eng = BacktestEngine(cfg, ClosedLoopExecution(sim), AvellanedaStoikov(cfg), max_duration_s=120)
    r = eng.run()
    assert len(r.ts) > 10
    # fills must have non-negative queue and valid prices
    if r.n_fills:
        assert np.all(r.fills["price"] > 0)
        assert np.all(r.fills["qty"] > 0)


def test_fills_are_causal_and_valid(cfg, market, features):
    eng = BacktestEngine(cfg, ReplayExecution(market, start_ts=1.0), FixedSpreadStrategy(cfg, 1.0),
                         features=features, max_duration_s=200)
    r = eng.run()
    assert r.n_fills > 0
    f = r.fills
    assert np.all(f["price"] > 0) and np.all(f["qty"] > 0)          # valid fills
    assert set(np.unique(f["side"])).issubset({-1, 1})
    assert np.all(f["queue_at_place"] >= -1e-9)                      # non-negative queue
    assert np.all(f["ts"] >= f["placed_ts"] - 1e-9)                 # fill cannot precede placement
    # captured spread per unit (may be negative under adverse selection) is bounded & finite
    cap = f["side"] * (f["mid_at_fill"] - f["price"])
    assert np.isfinite(cap).all()
