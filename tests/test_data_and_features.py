import numpy as np

from qamm.data import MarketData
from qamm.models import compute_features
from qamm.models.features import FEATURE_NAMES, OnlineFeatures


def test_market_data_roundtrip(market, tmp_path):
    p = market.save(tmp_path / "md.npz")
    md2 = MarketData.load(p)
    assert md2.book.n == market.book.n
    assert np.allclose(md2.book.bid_px, market.book.bid_px)
    assert md2.trades.n == market.trades.n


def test_book_invariants(market):
    b = market.book
    assert np.all(b.bid_px[:, 0] < b.ask_px[:, 0])            # uncrossed
    assert np.all(np.diff(b.bid_px, axis=1) < 1e-9)           # bids descending
    assert np.all(np.diff(b.ask_px, axis=1) > -1e-9)          # asks ascending
    assert np.all(b.ts[1:] >= b.ts[:-1])                      # time sorted


def test_features_finite(market, features):
    assert features.shape == (market.book.n, len(FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_imbalance_bounds(market):
    imb = market.book.imbalance(1)
    assert imb.min() >= -1.0 - 1e-9 and imb.max() <= 1.0 + 1e-9


def test_online_features_match_batch(market, features):
    """OnlineFeatures fed sequentially must reproduce the vectorised features (spread & imbalance exactly)."""
    b, tr = market.book, market.trades
    of = OnlineFeatures(market.tick_size)
    tp = 0
    checks = 0
    for i in range(min(b.n, 400)):
        ts = b.ts[i]
        new = []
        while tp < tr.n and tr.ts[tp] <= ts:
            new.append((tr.ts[tp], tr.side[tp], tr.size[tp])); tp += 1
        bs3 = b.bid_sz[i, :3].sum(); as3 = b.ask_sz[i, :3].sum()
        of.update(ts, b.bid_px[i, 0], b.bid_sz[i, 0], b.ask_px[i, 0], b.ask_sz[i, 0],
                  (bs3 - as3) / max(bs3 + as3, 1e-12), new)
        if i > 50:
            v = of.vector()
            assert abs(v[0] - features[i, 0]) < 1e-4      # spread ticks
            assert abs(v[1] - features[i, 1]) < 1e-4      # imbalance
            checks += 1
    assert checks > 100
