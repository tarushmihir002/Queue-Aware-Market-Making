"""Deterministic seeding across numpy / torch / python."""
from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int = 42) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
    return np.random.default_rng(seed)
