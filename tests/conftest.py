import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qamm.data import generate_synthetic
from qamm.models import compute_features
from qamm.utils import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def market(cfg):
    md = generate_synthetic(cfg, seed=5, duration_s=300)
    return md


@pytest.fixture(scope="session")
def features(market):
    return compute_features(market)
