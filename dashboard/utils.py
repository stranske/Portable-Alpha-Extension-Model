"""Shared dashboard utility helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from pa_core.config import ModelConfig, normalize_share

# Keep dashboard normalization aligned with core config behavior.


def build_alpha_shares_payload(
    active_share: float | None, theta_extpa: float | None
) -> dict[str, float] | None:
    """Return a normalized alpha-shares payload for session state."""
    if active_share is None or theta_extpa is None:
        return None
    active_share_norm = normalize_share(active_share)
    theta_extpa_norm = normalize_share(theta_extpa)
    if active_share_norm is None or theta_extpa_norm is None:
        return None
    return {
        "active_share": float(active_share_norm),
        "theta_extpa": float(theta_extpa_norm),
    }


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


def apply_promoted_alpha_shares(
    state: MutableMapping[str, Any],
    promoted_source: str | None,
    promoted_active_share: float | None,
    promoted_theta: float | None,
) -> bool:
    """Apply promoted alpha-share values and return True when a rerun is needed."""
    if promoted_active_share is None or promoted_theta is None:
        return False
    promoted_state = (promoted_source, promoted_active_share, promoted_theta)
    last_promoted = state.get("alpha_shares_last_promoted")
    promotion_token = state.get("scenario_grid_promotion_token")
    last_token = state.get("alpha_shares_last_promotion_token")
    if promoted_state == last_promoted and promotion_token == last_token:
        return False
    state["alpha_shares_active_share"] = promoted_active_share
    state["alpha_shares_theta_extpa"] = promoted_theta
    state["alpha_shares_last_promoted"] = promoted_state
    if promotion_token is not None:
        state["alpha_shares_last_promotion_token"] = promotion_token
        if promotion_token != last_token:
            state["portfolio_builder_autorun"] = True
            return True
    return False


__all__ = [
    "normalize_share",
    "build_alpha_shares_payload",
    "make_grid_cache_key",
    "bump_session_token",
    "apply_promoted_alpha_shares",
]
