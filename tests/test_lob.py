import numpy as np

from qamm.lob import OrderBook, Side


def test_queue_priority_market_order():
    b = OrderBook(tick_size=0.01, n_levels=5)
    b.add_limit(Side.ASK, 100.00, 10.0)      # public volume at best ask
    b.add_limit(Side.BID, 99.99, 10.0)
    o = b.place(Side.ASK, 100.00, 2.0, ts=0.0)   # our sell order joins behind the 10 public
    assert o.queue_ahead == 10.0
    # a buy market order of 5 consumes public only -> we are not filled, queue shrinks
    b.market_order(Side.BID, 5.0, ts=1.0)
    assert abs(o.queue_ahead - 5.0) < 1e-9 and o.remaining == 2.0
    # a buy of 6 more consumes remaining 5 public + 1 of ours
    b.market_order(Side.BID, 6.0, ts=2.0)
    assert o.queue_ahead == 0.0 and abs(o.remaining - 1.0) < 1e-9


def test_front_of_queue_fills_first():
    b = OrderBook(tick_size=0.01, n_levels=5)
    b.add_limit(Side.BID, 99.99, 5.0)
    o = b.place(Side.BID, 99.99, 3.0, ts=0.0)    # queue_ahead = 5
    b.add_limit(Side.BID, 99.99, 4.0)            # joins behind us; still ahead=5
    assert o.queue_ahead == 5.0
    b.market_order(Side.ASK, 7.0, ts=1.0)        # eats 5 ahead + 2 of ours
    assert abs(o.remaining - 1.0) < 1e-9


def test_public_cancel_reduces_queue_ahead():
    b = OrderBook(tick_size=0.01, n_levels=5)
    b.add_limit(Side.BID, 99.99, 10.0)
    o = b.place(Side.BID, 99.99, 2.0, ts=0.0)    # ahead = 10, level = 12
    b.cancel_public(Side.BID, 99.99, 5.0)        # cancel half the public volume
    assert o.queue_ahead < 10.0 and o.queue_ahead >= 0.0


def test_book_stays_uncrossed_and_sorted():
    b = OrderBook(tick_size=0.01, n_levels=5)
    for k in range(5):
        b.add_limit(Side.BID, 100.00 - (k + 1) * 0.01, 1.0)
        b.add_limit(Side.ASK, 100.00 + (k + 1) * 0.01, 1.0)
    assert b.best_bid < b.best_ask
    assert all(b.bids[i][0] > b.bids[i + 1][0] for i in range(len(b.bids) - 1))
    assert all(b.asks[i][0] < b.asks[i + 1][0] for i in range(len(b.asks) - 1))


def test_place_crossing_rejected():
    b = OrderBook(tick_size=0.01, n_levels=5)
    b.add_limit(Side.ASK, 100.00, 1.0)
    b.add_limit(Side.BID, 99.99, 1.0)
    try:
        b.place(Side.BID, 100.00, 1.0, ts=0.0)
        assert False
    except ValueError:
        pass
