The market problem

A market maker places limit orders to earn the bid–ask spread. But placing an order doesn't guarantee execution.

Three problems arise:

\- Your order may never get filled.

\- Your order may get filled precisely when the market is about to move against you.

\- You may lose queue priority and miss profitable executions.



A strategy that earns the spread on paper can lose money once adverse selection and realistic execution are included.



Your research question



Can we predict the probability that an order gets filled, estimate the adverse-selection cost conditional on a fill, and determine the optimal price and size at which to quote?



What you should build



A queue-aware market-making simulator with a probabilistic execution model.



Mathematical components:

1\. Estimate fill probability using survival analysis or competing-risk models.

2\. Model order arrivals and cancellations using Hawkes processes.

3\. Estimate adverse selection using conditional future mid-price movements.

4\. Model inventory risk and optimal quote placement using an Avellaneda–Stoikov-style framework.

5\. Compare a classical market maker against a Bayesian or reinforcement-learning-based policy.



A useful objective is:

max E\[{Spread Revenue}-{Adverse Selection}-{Fees}-{Inventory Cost}]



What makes your research different?



Don't simply predict whether the next candle is green or red.



Predict:

\- Probability of execution within 100 ms, 500 ms and 1 second.

\- Expected markout after execution.

\- Probability of cancellation before execution.

\- Expected P\&L conditional on queue position.

\- Optimal order size given current inventory and liquidity.



There is published research showing that queue position affects adverse-selection risk and inventory management. A 2025 Journal of Financial Markets paper also documents substantial effects of queue ordering on market depth. 

Link : [https://www.sciencedirect.com/science/article/pii/S1386418125000229?via%3Dihub](https://www.sciencedirect.com/science/article/pii/S1386418125000229?via%253Dihub)





Dataset: Binance BTCUSDT and ETHUSDT Level-2 order book and trade data it for NSE data as well



You can start with the public crypto-lob-stream dataset.



Technology stack



Python, NumPy, Polars, PyTorch, SciPy, survival analysis, Hawkes processes, C++ for the event-driven simulator.



Expected deliverable



A research paper, reproducible backtesting engine, and interactive market-making dashboard.



The most impressive result would be demonstrating that your model improves net realized P\&L after fees and adverse selection over a well-defined baseline.

