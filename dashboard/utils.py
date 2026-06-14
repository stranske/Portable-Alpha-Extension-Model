"""Shared dashboard utility helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from pa_core.config import ModelConfig, normalize_share
from pa_core.sleeve_suggestor import generate_sleeve_frontier, suggest_sleeve_sizes

# Keep dashboard normalization aligned with core config behavior.

DASHBOARD_ASSET_LIBRARY_KEY = "dashboard_asset_library_yaml"
DASHBOARD_PORTFOLIO_KEY = "dashboard_portfolio_yaml"
DASHBOARD_SCENARIO_KEY = "dashboard_active_scenario"
DASHBOARD_RESULT_KEY = "dashboard_active_result"

_CAPITAL_DEFAULTS = {
    "total_fund_capital": 1000.0,
    "external_pa_capital": 200.0,
    "active_ext_capital": 200.0,
    "internal_pa_capital": 200.0,
}

_SCENARIO_VALUE_FIELDS = (
    "analysis_mode",
    "n_simulations",
    "n_months",
    "total_fund_capital",
    "external_pa_capital",
    "active_ext_capital",
    "internal_pa_capital",
    "theta_extpa",
    "active_share",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_value(config: Any, field: str) -> Any:
    if isinstance(config, dict):
        if field in config:
            return config[field]
        upper_field = field.upper()
        if upper_field in config:
            return config[upper_field]
    value = getattr(config, field, None)
    if hasattr(value, "value"):
        return value.value
    return value


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def build_dashboard_scenario_payload(
    config: Any,
    *,
    source: str,
    yaml_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared in-session scenario payload used by dashboard pages."""
    values = {
        field: _config_value(config, field)
        for field in _SCENARIO_VALUE_FIELDS
        if _config_value(config, field) is not None
    }
    return {
        "source": source,
        "updated_at": _now_iso(),
        "values": values,
        "yaml_data": dict(yaml_data or {}),
    }


def store_dashboard_scenario(
    state: MutableMapping[str, Any],
    config: Any,
    *,
    source: str,
    yaml_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store the latest scenario in Streamlit session state."""
    payload = build_dashboard_scenario_payload(config, source=source, yaml_data=yaml_data)
    state[DASHBOARD_SCENARIO_KEY] = payload
    return payload


def store_dashboard_result(
    state: MutableMapping[str, Any],
    output_path: str,
    *,
    source: str,
    scenario: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Store the latest generated workbook path in Streamlit session state."""
    payload: dict[str, Any] = {
        "source": source,
        "updated_at": _now_iso(),
        "output_path": str(output_path),
    }
    if scenario is not None:
        payload["scenario"] = scenario
    if seed is not None:
        payload["seed"] = int(seed)
    state[DASHBOARD_RESULT_KEY] = payload
    return payload


def get_dashboard_result_path(state: MutableMapping[str, Any], default: str) -> str:
    """Return the most recent in-session result workbook path, if present."""
    result = state.get(DASHBOARD_RESULT_KEY)
    if isinstance(result, dict):
        output_path = result.get("output_path")
        if isinstance(output_path, str) and output_path.strip():
            return output_path
    return default


def get_dashboard_capital_defaults(
    state: MutableMapping[str, Any],
    fallback: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return capital defaults from the current session scenario/result."""
    defaults = dict(_CAPITAL_DEFAULTS)
    if fallback:
        defaults.update(fallback)

    scenario = state.get(DASHBOARD_SCENARIO_KEY)
    result = state.get(DASHBOARD_RESULT_KEY)
    if not isinstance(scenario, dict) and isinstance(result, dict):
        maybe_scenario = result.get("scenario")
        if isinstance(maybe_scenario, dict):
            scenario = maybe_scenario

    values = scenario.get("values") if isinstance(scenario, dict) else None
    if not isinstance(values, dict):
        return defaults

    return {
        key: _coerce_float(values.get(key), default_value)
        for key, default_value in defaults.items()
    }


def store_dashboard_asset_library(
    state: MutableMapping[str, Any],
    yaml_text: str,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Store calibrated asset-library YAML for Portfolio Builder handoff."""
    payload = {
        "source": "asset_library",
        "source_name": source_name,
        "updated_at": _now_iso(),
        "yaml": yaml_text,
    }
    state[DASHBOARD_ASSET_LIBRARY_KEY] = payload
    return payload


def get_dashboard_asset_library_yaml(state: MutableMapping[str, Any]) -> str | None:
    payload = state.get(DASHBOARD_ASSET_LIBRARY_KEY)
    if not isinstance(payload, dict):
        return None
    yaml_text = payload.get("yaml")
    return yaml_text if isinstance(yaml_text, str) and yaml_text.strip() else None


def store_dashboard_portfolio(
    state: MutableMapping[str, Any],
    yaml_text: str,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Store generated portfolio YAML so file export is not the only handoff."""
    payload = {
        "source": "portfolio_builder",
        "source_name": source_name,
        "updated_at": _now_iso(),
        "yaml": yaml_text,
    }
    state[DASHBOARD_PORTFOLIO_KEY] = payload
    return payload


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


def run_sleeve_suggestions(
    cfg: ModelConfig,
    index_series: pd.Series,
    *,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    max_shortfall: float,
    step: float,
    max_evals: int,
    constraint_scope: str,
    seed: int | None,
) -> pd.DataFrame:
    """Run the sleeve suggestor with a shared seed."""
    return suggest_sleeve_sizes(
        cfg,
        index_series,
        max_te=max_te,
        max_breach=max_breach,
        max_cvar=max_cvar,
        max_shortfall=max_shortfall,
        step=step,
        max_evals=max_evals,
        constraint_scope=constraint_scope,
        seed=seed,
    )


def run_sleeve_frontier(
    cfg: ModelConfig,
    index_series: pd.Series,
    *,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    max_shortfall: float,
    step: float,
    max_evals: int,
    constraint_scope: str,
    seed: int | None,
) -> pd.DataFrame:
    """Run the sleeve frontier with a shared seed."""
    return generate_sleeve_frontier(
        cfg,
        index_series,
        max_te=max_te,
        max_breach=max_breach,
        max_cvar=max_cvar,
        max_shortfall=max_shortfall,
        step=step,
        max_evals=max_evals,
        constraint_scope=constraint_scope,
        seed=seed,
    )


__all__ = [
    "normalize_share",
    "DASHBOARD_ASSET_LIBRARY_KEY",
    "DASHBOARD_PORTFOLIO_KEY",
    "DASHBOARD_SCENARIO_KEY",
    "DASHBOARD_RESULT_KEY",
    "build_dashboard_scenario_payload",
    "store_dashboard_scenario",
    "store_dashboard_result",
    "get_dashboard_result_path",
    "get_dashboard_capital_defaults",
    "store_dashboard_asset_library",
    "get_dashboard_asset_library_yaml",
    "store_dashboard_portfolio",
    "build_alpha_shares_payload",
    "make_grid_cache_key",
    "bump_session_token",
    "apply_promoted_alpha_shares",
    "run_sleeve_suggestions",
    "run_sleeve_frontier",
]
