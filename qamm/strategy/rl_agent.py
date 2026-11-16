"""
Double-DQN agent (PyTorch) for the MarketMakingEnv, and `RLStrategy` which
plugs the trained policy into the BacktestEngine like any other strategy.
"""
from __future__ import annotations

import time
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..backtest.engine import MarketState, QuoteDecision
from ..models.features import FEATURE_NAMES
from .base import Strategy
from .rl_env import ACTION_TABLE, OBS_DIM, MarketMakingEnv


class QNet(nn.Module):
    def __init__(self, d_in: int, n_act: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_act))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int = OBS_DIM, n_act: int = len(ACTION_TABLE), hidden: int = 128, lr: float = 5e-4,
                 gamma: float = 0.99, buffer_size: int = 100_000, batch_size: int = 256, target_update: int = 500,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay_steps: int = 30_000, seed: int = 0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = QNet(obs_dim, n_act, hidden)
        self.qt = QNet(obs_dim, n_act, hidden)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.bs, self.target_update = gamma, batch_size, target_update
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_steps
        self.buf = deque(maxlen=buffer_size)
        self.n_act = n_act
        self.steps = 0
        self.obs_mu = np.zeros(obs_dim, dtype=np.float32)
        self.obs_sd = np.ones(obs_dim, dtype=np.float32)
        self.history: List[dict] = []

    def epsilon(self) -> float:
        f = min(self.steps / self.eps_decay, 1.0)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def _norm(self, o: np.ndarray) -> np.ndarray:
        return (o - self.obs_mu) / self.obs_sd

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_act))
        with torch.no_grad():
            q = self.q(torch.from_numpy(self._norm(obs)[None]))
        return int(torch.argmax(q, dim=1).item())

    def remember(self, o, a, r, o2, done):
        self.buf.append((o, a, r, o2, done))

    def update(self) -> Optional[float]:
        if len(self.buf) < self.bs:
            return None
        idx = self.rng.choice(len(self.buf), self.bs, replace=False)
        batch = [self.buf[i] for i in idx]
        o = torch.from_numpy(np.stack([self._norm(b[0]) for b in batch]))
        a = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        o2 = torch.from_numpy(np.stack([self._norm(b[3]) for b in batch]))
        d = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            a2 = torch.argmax(self.q(o2), dim=1, keepdim=True)        # double DQN
            q2 = self.qt(o2).gather(1, a2).squeeze(1)
            target = r + self.gamma * (1 - d) * q2
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.qt.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save({"q": self.q.state_dict(), "mu": self.obs_mu, "sd": self.obs_sd, "history": self.history,
                    "steps": self.steps}, path)

    @classmethod
    def load(cls, path, **kw) -> "DQNAgent":
        ck = torch.load(path, weights_only=False)
        ag = cls(**kw)
        ag.q.load_state_dict(ck["q"]); ag.qt.load_state_dict(ck["q"])
        ag.obs_mu, ag.obs_sd, ag.history, ag.steps = ck["mu"], ck["sd"], ck["history"], ck["steps"]
        return ag


def train_dqn(env: MarketMakingEnv, agent: DQNAgent, episodes: int = 50, warmup_episodes: int = 2,
              verbose: bool = True, log_every: int = 1) -> DQNAgent:
    """Train with ε-greedy; the first `warmup_episodes` only fill the buffer and set obs normalisation."""
    t0 = time.time()
    obs_buf = []
    for ep in range(episodes):
        o, _ = env.reset()
        ep_r = 0.0; losses = []
        done = False
        while not done:
            a = agent.act(o) if ep >= warmup_episodes else int(agent.rng.integers(agent.n_act))
            o2, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.remember(o, a, r, o2, float(done))
            if ep < warmup_episodes:
                obs_buf.append(o)
            else:
                l = agent.update()
                if l is not None:
                    losses.append(l)
            ep_r += r
            o = o2
        if ep == warmup_episodes - 1 and obs_buf:
            arr = np.stack(obs_buf)
            agent.obs_mu = arr.mean(0).astype(np.float32)
            agent.obs_sd = (arr.std(0) + 1e-6).astype(np.float32)
        rec = {"episode": ep, "reward": ep_r, "pnl": info["pnl"], "fills": info["fills"], "inv_end": info["inv"],
               "eps": agent.epsilon(), "loss": float(np.mean(losses)) if losses else float("nan"),
               "wall_s": time.time() - t0}
        agent.history.append(rec)
        if verbose and ep % log_every == 0:
            print(f"  ep {ep:3d}  reward {ep_r:9.2f}  pnl {info['pnl']:+.4f}  fills {info['fills']:4d}  "
                  f"inv_end {info['inv']:+.1f}  eps {agent.epsilon():.2f}  loss {rec['loss']:.4f}  "
                  f"[{time.time() - t0:.0f}s]")
    return agent


class RLStrategy(Strategy):
    """Greedy policy from a trained DQNAgent, usable in BacktestEngine."""
    name = "rl_dqn"

    def __init__(self, cfg: dict, agent: DQNAgent):
        super().__init__(cfg)
        self.agent = agent
        self._last_flow = 0.0

    def reset(self):
        self._last_flow = 0.0

    def _obs(self, s: MarketState) -> np.ndarray:
        bq = aq = -1.0
        for o in s.open_orders.values():
            top = s.bid_sz1 if o.side > 0 else s.ask_sz1
            r = o.queue_ahead / max(top, 1e-9)
            if o.side > 0: bq = r
            else: aq = r
        inv = s.inventory / self.max_inv
        return np.concatenate([s.features, [inv, inv * inv, s.t_frac, bq, aq, self._last_flow]]).astype(np.float32)

    def decide(self, s: MarketState) -> QuoteDecision:
        a = self.agent.act(self._obs(s), greedy=True)
        bd, ad, m = ACTION_TABLE[a]
        dec = QuoteDecision()
        if bd is not None:
            dec.bid_price, dec.bid_size = s.best_bid - bd * s.tick, m * self.size
        if ad is not None:
            dec.ask_price, dec.ask_size = s.best_ask + ad * s.tick, m * self.size
        return dec

    def on_step_end(self, state, fills, trades):
        self._last_flow = sum(f.side * f.qty for f in fills) / self.size
