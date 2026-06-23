"""Shared dashboard utility helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import MutableMapping
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from pa_core.config import ModelConfig, normalize_share
from pa_core.data import load_index_returns
from pa_core.sleeve_suggestor import generate_sleeve_frontier, suggest_sleeve_sizes

# Keep dashboard normalization aligned with core config behavior.

RESULTS_EMPTY_STATE_MESSAGE = (
    "No results yet - complete a run in the Scenario Wizard, Scenario Grid, or Stress Lab "
    "to generate output."
)
RUN_LOGS_EMPTY_STATE_MESSAGE = "No runs yet - run a scenario and its history will appear here."

CURRENT_SCENARIO_CONFIG_KEY = "dashboard_current_scenario_config"
CURRENT_RESULTS_PATH_KEY = "dashboard_current_results_path"
CURRENT_INDEX_RETURNS_KEY = "dashboard_current_index_returns"

_CONFIG_FIELD_LABELS = {
    "financing_mode": "Financing draw mode",
    "financing_model": "Financing model",
    "financing_schedule_path": "Financing schedule file",
    "N_SIMULATIONS": "Number of simulations",
    "Number of simulations": "Number of simulations",
    "N_MONTHS": "Number of months",
    "Number of months": "Number of months",
    "risk_metrics": "Risk metrics",
    "total_fund_capital": "Total fund capital",
    "external_pa_capital": "External PA capital",
    "active_ext_capital": "Active extension capital",
    "internal_pa_capital": "Internal PA capital",
}


def friendly_validation_error_message(exc: BaseException) -> str:
    """Return an operator-facing message for config validation failures."""

    validation_exc = exc.__cause__ if exc.__cause__ is not None else exc
    fields: list[str] = []
    errors = getattr(validation_exc, "errors", None)
    if callable(errors):
        for error in errors():
            loc = error.get("loc", ())
            field = str(loc[-1]) if loc else ""
            label = _CONFIG_FIELD_LABELS.get(field)
            if label and label not in fields:
                fields.append(label)

    raw = str(exc)
    for field, label in _CONFIG_FIELD_LABELS.items():
        if field in raw and label not in fields:
            fields.append(label)

    if not fields:
        return (
            "The scenario settings need a correction before the dashboard can run them. "
            "Review the highlighted inputs and try again."
        )

    field_list = ", ".join(fields)
    return (
        f"The scenario settings need a correction before the dashboard can run them: "
        f"{field_list}. Choose or update the highlighted inputs, then try again."
    )


def remember_current_scenario(
    state: MutableMapping[str, Any],
    *,
    config: ModelConfig,
    results_path: str | Path | None = None,
    index_returns: pd.Series | None = None,
) -> None:
    """Persist the latest runnable scenario in session state for other pages."""
    state[CURRENT_SCENARIO_CONFIG_KEY] = config
    if results_path is not None:
        state[CURRENT_RESULTS_PATH_KEY] = str(results_path)
    if index_returns is not None:
        state[CURRENT_INDEX_RETURNS_KEY] = index_returns.copy()


def current_scenario_config(state: MutableMapping[str, Any]) -> ModelConfig | None:
    """Return the latest in-session scenario config when available."""
    config = state.get(CURRENT_SCENARIO_CONFIG_KEY)
    return config if isinstance(config, ModelConfig) else None


def current_results_path(state: MutableMapping[str, Any]) -> str | None:
    """Return the latest results workbook path remembered by the wizard."""
    raw_path = state.get(CURRENT_RESULTS_PATH_KEY)
    if isinstance(raw_path, Path):
        return str(raw_path)
    if isinstance(raw_path, str) and raw_path.strip():
        return raw_path
    return None


def current_index_returns(state: MutableMapping[str, Any]) -> pd.Series | None:
    """Return a copy of the latest uploaded index returns from session state."""
    series = state.get(CURRENT_INDEX_RETURNS_KEY)
    if isinstance(series, pd.Series):
        return series.copy()
    return None


def config_capital_defaults(config: ModelConfig | None) -> dict[str, float]:
    """Return Stress Lab-compatible capital defaults from a stored config."""
    if config is None:
        return {
            "total_fund_capital": 1000.0,
            "external_pa_capital": 200.0,
            "active_ext_capital": 200.0,
            "internal_pa_capital": 200.0,
        }
    return {
        "total_fund_capital": float(config.total_fund_capital),
        "external_pa_capital": float(config.external_pa_capital),
        "active_ext_capital": float(config.active_ext_capital),
        "internal_pa_capital": float(config.internal_pa_capital),
    }


# Bundled index-returns dataset shipped with the repo so first-run users can run
# an end-to-end example without uploading their own file (see issue #1900).
SAMPLE_INDEX_FILENAME = "sp500tr_fred_divyield.csv"
SAMPLE_FINANCING_MODE = "per_path"


def bundled_sample_index_path() -> Path:
    """Return the path to the bundled sample index-returns CSV.

    The file is shipped as package data so installed ``pa-dashboard`` users get
    the same sample series as repo-checkout users. Returning a path keeps
    callers free to stream it into a download button or read it lazily only when
    the user opts in.
    """
    resource = resources.files("data").joinpath(SAMPLE_INDEX_FILENAME)
    with resources.as_file(resource) as path:
        return path


def load_bundled_sample_index() -> pd.Series:
    """Return the bundled sample index returns as a numeric ``pd.Series``.

    Uses the core index-return loader so the one-click sample follows the same
    ``Monthly_TR`` column and date-sorting rules as CLI and wizard simulations.
    """
    path = bundled_sample_index_path()
    return load_index_returns(path)


def build_sample_model_config(**overrides: Any) -> ModelConfig:
    """Return a ModelConfig for bundled-sample dashboard paths."""
    data: dict[str, Any] = {
        "Number of simulations": 1000,
        "Number of months": 12,
        "financing_mode": SAMPLE_FINANCING_MODE,
    }
    data.update(overrides)
    return ModelConfig.model_validate(data)


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
    "CURRENT_SCENARIO_CONFIG_KEY",
    "CURRENT_RESULTS_PATH_KEY",
    "CURRENT_INDEX_RETURNS_KEY",
    "remember_current_scenario",
    "current_scenario_config",
    "current_results_path",
    "current_index_returns",
    "config_capital_defaults",
    "build_alpha_shares_payload",
    "make_grid_cache_key",
    "bump_session_token",
    "apply_promoted_alpha_shares",
    "run_sleeve_suggestions",
    "run_sleeve_frontier",
    "SAMPLE_INDEX_FILENAME",
    "bundled_sample_index_path",
    "load_bundled_sample_index",
]
