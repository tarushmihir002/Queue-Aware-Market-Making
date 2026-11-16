import numpy as np

from qamm.models import compute_features
from qamm.strategy import ACTION_TABLE, DQNAgent, MarketMakingEnv, train_dqn
from qamm.strategy.rl_env import OBS_DIM


def test_env_step_contract(cfg, market, features):
    env = MarketMakingEnv(cfg, [market], [features], episode_len_steps=100, seed=0, random_start=False)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
    total = 0.0
    for _ in range(50):
        a = env.action_space.sample() if hasattr(env, "action_space") else np.random.randint(len(ACTION_TABLE))
        obs, r, term, trunc, info = env.step(int(a))
        assert np.isfinite(obs).all() and np.isfinite(r)
        total += r
        if term or trunc:
            break
    assert "pnl" in info and "inv" in info


def test_dqn_trains_a_few_episodes(cfg, market, features):
    env = MarketMakingEnv(cfg, [market], [features], episode_len_steps=150, seed=0)
    ag = DQNAgent(seed=0, eps_decay_steps=2000, batch_size=64, buffer_size=5000)
    train_dqn(env, ag, episodes=4, warmup_episodes=1, verbose=False)
    assert len(ag.history) == 4
    # greedy action is a valid index
    obs, _ = env.reset()
    a = ag.act(obs, greedy=True)
    assert 0 <= a < len(ACTION_TABLE)
