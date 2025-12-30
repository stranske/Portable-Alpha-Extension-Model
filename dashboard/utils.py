"""Shared dashboard utility helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from pa_core.config import ModelConfig


def normalize_share(value: float | None) -> float | None:
    """Normalize percentage-style inputs to a 0..1 fraction."""
    if value is None:
        return None
    if value > 1.0:
        return value / 100.0
    return value


_GRID_CACHE_VERSION = 1


def make_grid_cache_key(
    cfg: ModelConfig, index_series: pd.Series, seed: int, y_axis_mode: str
) -> str:
    """Return a stable cache key for scenario grid results."""
    cfg_json = json.dumps(cfg.model_dump(), sort_keys=True)
    hash_fn = getattr(pd, "util").hash_pandas_object  # type: ignore[attr-defined]
    idx_hash = hashlib.sha256(hash_fn(index_series).values.tobytes()).hexdigest()
    payload = json.dumps(
        {
            "cfg": cfg_json,
            "idx": idx_hash,
            "seed": seed,
            "y_axis": y_axis_mode,
            "v": _GRID_CACHE_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def bump_session_token(state: MutableMapping[str, Any], key: str) -> int:
    """Increment a session-state token used for cross-page reruns."""
    current = state.get(key, 0)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        next_token = 1
    else:
        next_token = int(current) + 1
    state[key] = next_token
    return next_token


__all__ = ["normalize_share", "make_grid_cache_key", "bump_session_token"]
