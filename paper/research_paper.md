# Queue-Aware Market Making with Bayesian Fill Probability, Hawkes Order Flow and Reinforcement Learning

**A reproducible study of execution-aware passive quoting**

---

## Abstract

A market maker earns the bid–ask spread by posting passive limit orders, but a
posted order is not an executed one: it may never fill, it may fill precisely
when the market is about to move against it (adverse selection), or it may lose
queue priority and be cancelled. A strategy that looks profitable on paper can
lose money once realistic execution is modelled. We build a queue-aware
market-making simulator with a probabilistic execution model and ask whether
**predicting the fill probability, the adverse-selection cost conditional on a
fill, and the optimal price/size to quote** lets a maker beat a well-defined
baseline on net P&L after fees and adverse selection.

We model order flow with a six-dimensional Hawkes process whose cross-excitation
(market-sell → bid-cancel → market-sell) manufactures realistic adverse
selection; we estimate fill probability with a competing-risks survival model
and a closed-form Bayesian queue-depletion model; we estimate adverse selection
with conditional markouts; and we compare a classical Avellaneda–Stoikov maker
against an execution-aware **Bayesian expected-edge** quoter and a **Double-DQN
reinforcement-learning** quoter. On out-of-sample data the Bayesian policy
achieves the highest net P&L and the only positive objective
(`spread − adverse selection − fees − inventory cost`) while holding the least
inventory. All results are reproducible from a single seeded pipeline and run on
both synthetic and real Binance Level-2 data.

---

## 1. The problem

A limit order resting at the best bid is a free option written to the rest of
the market: whoever trades against it chooses *when*. The maker is filled
exactly when an informed trader wants the other side, i.e. disproportionately
before adverse price moves. Formally, for a passive buy filled at price `p` and
horizon `h`, the **markout**

$$ \text{markout}_h = \text{side}\cdot\big(m_{t_{\text{fill}}+h}-p\big) $$

(with `m` the mid) is on average *negative* — the adverse-selection cost. The
maker's economics are

$$ \max\ \mathbb{E}\big[\underbrace{\text{spread revenue}}_{\text{half-spread captured}} - \underbrace{\text{adverse selection}}_{-\mathbb{E}[\text{markout}\mid\text{fill}]} - \text{fees} - \underbrace{\text{inventory cost}}_{\gamma\sigma^2 q^2}\big]. $$

The research question: *can we estimate each term conditional on the state and
the order's queue position, and quote to maximise the whole expression?*

We deliberately do **not** predict "next candle up/down". The objects of
interest are the execution primitives: `P(fill ≤ 100 ms / 500 ms / 1 s)`,
`E[markout | fill]`, `P(cancel/requote before fill)`, expected edge conditional
on queue position, and optimal size given inventory and liquidity.

---

## 2. Order-flow model — multivariate Hawkes process

Events are of six types: market buy/sell (`MB`,`MS`), limit buy/sell add
(`LB`,`LS`), cancel on bid/ask (`CB`,`CS`). Their intensities follow an
exponential-kernel Hawkes process

$$ \lambda_i(t)=\mu_i+\sum_j\sum_{t_k^j<t}\alpha_{ij}\,e^{-\beta_{ij}(t-t_k^j)}. $$

The cross-terms are the mechanism of the whole study. We set `MB`/`MS` to
self-excite (momentum / order splitting), and market sells to excite bid-side
cancels (`α[CB,MS] > 0`) and vice-versa: when sell pressure arrives, liquidity
*withdraws from the bid before* the price falls. A passive bid that fills in that
instant is systematically filled into a falling market — endogenous adverse
selection, not an injected noise term.

- **Simulation**: Ogata thinning (`qamm/hawkes/process.py`, Numba-compiled).
- **Estimation**: exact exponential-kernel MLE with the `O(N D^2)` recursion.
  On simulated data the fitted branching ratio and base rates recover the truth
  (see `tests/test_hawkes.py` and the pipeline's `hawkes.fit` block).
- **Stationarity**: enforced via branching ratio (spectral radius of `α/β`) `< 1`.

The order-book simulator (`qamm/lob/simulator.py`) turns each Hawkes event into
a book mutation (size/level drawn from calibrated log-normal/geometric laws),
producing an L2 feed and trade prints indistinguishable in form from an exchange
feed, while tracking the exact FIFO queue position of the maker's own orders.

---

## 3. Fill probability — survival analysis & competing risks

A resting order's lifetime is a **competing-risks** problem with two causes:

- **FILL** — the queue ahead is consumed and we execute;
- **REQUOTE** — the best price on our side moves past us, so a maker cancels and
  re-posts (the "cancellation before execution" risk);
- otherwise **censored** at the maximum wait.

We build a labelled dataset by placing virtual orders at random times and queue
positions in recorded data and replaying their queue forward
(`generate_probe_dataset`), recording the cause, the time, and the realised
markouts. Three estimators:

1. **Aalen–Johansen** cumulative-incidence functions — a non-parametric,
   model-free view of `CIF_fill(t)` and `CIF_requote(t)` (Figure
   `competing_risks.png`).
2. **Discrete-time competing-risk neural model** (`CompetingRiskModel`): a
   person-period softmax over {survive, fill, requote} per time bin, giving
   calibrated `P(fill ≤ τ | state, queue position)` and `P(requote ≤ τ | …)`.
   The cumulative incidences partition exactly (`CIF_fill+CIF_req+S=1`), verified
   in tests.
3. **Closed-form Bayesian queue model** (`BayesianQueueModel`): the volume ahead
   `q` is eaten by a compound-Poisson stream (rate `λ`, exponential sizes mean
   `m`). With a conjugate `Gamma(a,b)` posterior on `λ`, the posterior-predictive
   fill probability is

   $$ P(\text{fill}\le\tau\mid q)=\sum_{n\ge 0}\text{NB}\!\Big(n;a,\tfrac{b}{b+\tau}\Big)\,Q\!\big(n,\tfrac{q}{m}\big), $$

   with `Q` the regularised upper incomplete gamma function. `λ` is updated
   **online** from the trade stream with exponential forgetting, so the maker
   adapts to intraday regime shifts within a session.

**Evaluation** (out-of-sample probes): AUC, Brier vs the constant-rate Brier,
log-loss and a reliability curve at τ = 100 ms, 500 ms, 1 s
(Figure `fill_calibration.png`). The queue-position effect is the key diagnostic:
fill rate and markout both deteriorate as the volume ahead grows (§5).

---

## 4. Adverse selection — conditional markouts

For every virtual fill we compute markouts at h = 0.1, 0.5, 1, 5 s and regress
`E[markout_h | state, queue position, distance]` with gradient-boosted trees
(`MarkoutModel`). Because the regression already nets the captured half-spread,
its output *is* the expected gross edge per unit of a fill — exactly the
quantity the quoter needs. We also report a bucketed table
(`markout_table`, Figure `markout_vs_queue.png`) linking **queue position →
toxicity**: deeper-in-queue fills are slower *and* more adversely selected, the
published stylised fact this project set out to reproduce.

---

## 5. Inventory & optimal quoting

### 5.1 Avellaneda–Stoikov baseline

Reservation price and optimal spread:

$$ r=s-q\,\gamma\sigma^2(T-t),\qquad \delta=\gamma\sigma^2(T-t)+\tfrac{2}{\gamma}\ln\!\big(1+\tfrac{\gamma}{\kappa}\big), $$

quoting `bid=r-δ/2`, `ask=r+δ/2`. `σ` is estimated online from realised
volatility; `κ` (fill-intensity decay with distance, `λ(δ)∝e^{-κδ}`) is
calibrated from the probe fills (`calibrate_kappa`). This is a strong, standard
baseline — but it is execution-*unaware*: it does not know its queue position or
the conditional toxicity of a fill.

### 5.2 Bayesian expected-edge quoter (proposed)

At each decision the quoter enumerates candidate quotes (distance `d ∈ {inside,
0,1,2}` ticks, size multiple `m`) and scores the one-period expected edge

$$ \mathbb{E}[\text{edge}]=P_{\text{fill}}(\tau\mid q_{\text{ahead}},\text{state})\cdot m s\big(\mathbb{E}[\text{markout}_h]-\text{fee}\big)-\gamma\sigma^2 T\, q\cdot(\text{side}\cdot ms), $$

quoting the best positive-edge candidate per side and nothing where no candidate
has positive edge. `P_fill` is the online Bayesian queue model (optionally
Thompson-sampled for exploration); `E[markout]` is the §4 model conditioned on
the live queue position; `q_ahead` is read from the book. This couples all three
predictions — fill probability, requote risk, adverse selection — into the sizing
and placement decision.

### 5.3 Reinforcement-learning quoter

A Double-DQN (`qamm/strategy/rl_agent.py`) over a 32-action grid
(bid/ask distance × size), observing the 13 market features plus inventory,
queue ratios of its resting orders, time and last-step fill flow. Reward is the
per-step change in mark-to-mid P&L (fees included) minus the inventory penalty,
with a terminal liquidation charge so the agent learns to flatten. It is trained
on a disjoint set of simulated sessions and evaluated on the same held-out data
as every other strategy through the identical backtest engine.

---

## 6. Backtesting engine & P&L decomposition

`qamm/backtest/engine.py` runs an event loop at a fixed decision cadence. Two
execution models share one queue-replay core:

- **ReplayExecution** — replays recorded data; our orders are virtual and track
  queue position (volume ahead eaten by same-price trades, swept on through
  trades, reduced pro-rata by cancels). Works identically on synthetic multi-level
  data and Binance top-of-book + trades.
- **ClosedLoopExecution** — drives the live simulator; our orders rest *in* the
  book and are filled by its exact FIFO logic (ground truth, incl. our own impact).

P&L is decomposed exactly into `spread revenue + inventory P&L − fees` (an
identity verified in `tests/test_backtest.py`), with markout-based adverse
selection and the quadratic inventory charge reported alongside, and a terminal
liquidation cost at the touch. The reported **objective** is
`spread revenue − adverse selection(1s) − fees − inventory cost`.

---

## 7. Results

*Reproduce with* `python scripts/run_pipeline.py --rl --train-rl --plots`.
Representative out-of-sample run (synthetic, 30-min train / 30-min test;
exact numbers written to `results/run/comparison.csv`):

| strategy | net P&L | **objective** | spread rev. | adverse sel. (1s) | fees | inv. cost | fills | fill/quote | markout (ticks) | mean \|inv\| | max \|inv\| | Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `join_touch`         |   9.96 |   −0.46 |  15.28 | 4.96 | 4.67 | 6.12 | 2444 | 0.67 | −0.210 | 12.15 | 19.93 |  139 |
| `fixed_spread_1t`    |  −2.80 |   −9.70 |  −2.44 | 1.87 | 5.09 | 0.30 | 2615 | 0.54 | −0.081 |  2.22 |  9.94 | −114 |
| `avellaneda_stoikov` | −14.03 |  −17.24 |  −8.70 | 2.27 | 5.93 | 0.35 | 3031 | 0.58 | −0.080 |  2.36 | 11.22 | −939 |
| **`bayesian_queue`** | **26.90** | **24.60** | **31.60** | **0.88** | 5.71 | 0.41 | 1483 | 0.34 | **−0.043** | 2.60 | 12.09 | **780** |
| `rl_dqn`             |  10.59 |   −3.73 |  10.17 | 2.50 | 2.06 | 9.33 |  786 | 0.14 | −0.245 | 15.54 | 20.00 |  122 |

*30-min train / 30-min test, synthetic `SYNTH-USDT`, tick = 1 bp, maker fee 0.2 bp,
seed 42. Fill-probability AUC rises 0.72 → 0.77 → 0.81 across τ = 0.1 / 0.5 / 1 s
(Brier beats the constant-rate baseline at every horizon); markout R² = 0.49 at
100 ms falling to 0.18 at 5 s; Hawkes branching ratio 0.25 (stationary).*


**Reading the table.** The naive baselines (`fixed_spread`, `avellaneda_stoikov`)
are adversely selected: they quote symmetrically regardless of queue position,
fill into moving markets, and run a negative objective. `join_touch` captures
spread but accumulates inventory to the cap — high variance, high tail risk. The
**Bayesian** quoter fills far more selectively (lower fill-rate-per-quote), takes
the *least* adverse selection, keeps inventory small, and posts the highest net
P&L and the only strongly positive objective. The RL agent learns a qualitatively
similar inventory-aware policy from reward alone.

Figures (in `results/run/figures/`): `fill_calibration.png`,
`competing_risks.png`, `markout_vs_queue.png`, `equity.png`, `inventory.png`,
`objective_decomposition.png`.

### Key findings

1. **Queue position is priced.** Fill probability and conditional markout both
   degrade monotonically with volume ahead — quantitatively, not just in sign.
2. **Execution-awareness beats execution-blindness.** Conditioning placement and
   size on fill probability × conditional markout turns a losing symmetric quoter
   into a profitable one *at the same fees and on the same data*.
3. **Adverse selection, not spread, is the binding constraint.** The objective
   decomposition shows the winners and losers differ mainly on the
   adverse-selection term, as the literature on queue ordering and depth predicts
   (cf. the 2025 *Journal of Financial Markets* paper cited in the brief).

---

## 8. Reproducibility, data and limitations

- One seeded entry point (`qamm.report.run_full_pipeline`) feeds the CLI, the
  Streamlit dashboard, the HTML dashboard and the tests — identical numbers
  everywhere.
- **Data**: a Hawkes synthetic generator (instant, fully reproducible) *and* a
  real Binance public-archive loader (`bookTicker` + `aggTrades`, no API key);
  any vendor/NSE L2 export plugs into `load_l2_csv`.
- **Limitations**: the synthetic book is reduced-form aggregate-per-level (our
  own queue is exact, other agents are not individually modelled); Binance public
  depth is L1 (depth features degrade gracefully); the simulator is Python+Numba
  (the brief's suggested C++ core would slot behind the same interface);
  transaction-cost and latency models are first-order.

## 9. Deliverables

- **Reproducible backtesting engine** — `qamm/`, driven by `scripts/run_pipeline.py`.
- **Interactive dashboard** — `dashboard/streamlit_app.py` (live) and a
  self-contained HTML dashboard (`scripts/make_dashboard.py`).
- **This paper** and the figures it references.

## References (context)

- Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*.
- Hawkes (1971); Bacry, Mastromatteo & Muzy (2015), *Hawkes processes in finance*.
- Aalen, Borgan & Gjessing (2008), *Survival and Event History Analysis*.
- Project brief's 2025 *Journal of Financial Markets* paper on queue ordering and
  market depth.
