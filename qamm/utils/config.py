"""Tiny YAML config loader with attribute access."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class Config(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(dict(self)))


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load YAML config (default: config/default.yaml). Nested overrides via dotted keys,
    e.g. load_config(**{"strategy.gamma": 0.2})."""
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config(raw)
    for k, v in overrides.items():
        node: dict = cfg
        parts = k.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v
    return cfg
