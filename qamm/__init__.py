"""
qamm — Queue-Aware Market Making.

A research framework for market making with:
  * Hawkes-process order-flow modelling           (qamm.hawkes)
  * An event-driven queue-tracking limit order book (qamm.lob)
  * Bayesian / survival-analysis fill probability   (qamm.models.survival)
  * Adverse-selection (markout) estimation          (qamm.models.adverse_selection)
  * Avellaneda–Stoikov, Bayesian and RL quoting     (qamm.strategy)
  * A reproducible backtesting engine               (qamm.backtest)
"""

__version__ = "0.1.0"
