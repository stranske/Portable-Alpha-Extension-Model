from __future__ import annotations

import pandas as pd

from dashboard.utils import (
    apply_promoted_alpha_shares,
    build_alpha_shares_payload,
    bump_session_token,
    make_grid_cache_key,
    normalize_share,
)
from pa_core.config import ModelConfig


def test_normalize_share_handles_percent_values() -> None:
    assert normalize_share(60.0) == 0.6


def test_normalize_share_preserves_fraction_values() -> None:
    assert normalize_share(0.6) == 0.6


def test_normalize_share_handles_none() -> None:
    assert normalize_share(None) is None


def test_build_alpha_shares_payload_normalizes_values() -> None:
    payload = build_alpha_shares_payload(20.0, 30.0)
    assert payload == {"active_share": 0.2, "theta_extpa": 0.3}


def test_build_alpha_shares_payload_handles_none() -> None:
    assert build_alpha_shares_payload(None, 0.3) is None


def test_make_grid_cache_key_stable() -> None:
    cfg = ModelConfig.model_validate(
        {
            "Number of simulations": 1000,
            "Number of months": 12,
            "analysis_mode": "alpha_shares",
        }
    )
    idx = pd.Series([0.01, 0.02, 0.03])
    key1 = make_grid_cache_key(cfg, idx, 42, "External alpha fraction (theta)")
    key2 = make_grid_cache_key(cfg, idx, 42, "External alpha fraction (theta)")
    assert key1 == key2


def test_make_grid_cache_key_changes_with_axis() -> None:
    cfg = ModelConfig.model_validate(
        {
            "Number of simulations": 1000,
            "Number of months": 12,
            "analysis_mode": "alpha_shares",
        }
    )
    idx = pd.Series([0.01, 0.02, 0.03])
    key1 = make_grid_cache_key(cfg, idx, 42, "External alpha fraction (theta)")
    key2 = make_grid_cache_key(cfg, idx, 42, "External PA $ (mm)")
    assert key1 != key2


def test_bump_session_token_increments_int_values() -> None:
    state: dict[str, object] = {"token": 2}
    assert bump_session_token(state, "token") == 3
    assert state["token"] == 3


def test_bump_session_token_resets_invalid_values() -> None:
    state: dict[str, object] = {"token": "abc"}
    assert bump_session_token(state, "token") == 1
    assert state["token"] == 1


def test_apply_promoted_alpha_shares_triggers_rerun_on_new_token() -> None:
    state: dict[str, object] = {
        "scenario_grid_promotion_token": 2,
        "alpha_shares_last_promotion_token": 1,
    }
    rerun = apply_promoted_alpha_shares(state, "Scenario Grid", 0.4, 0.3)
    assert rerun is True
    assert state["alpha_shares_active_share"] == 0.4
    assert state["alpha_shares_theta_extpa"] == 0.3
    assert state["alpha_shares_last_promotion_token"] == 2
    assert state["portfolio_builder_autorun"] is True


def test_apply_promoted_alpha_shares_skips_when_unchanged() -> None:
    state: dict[str, object] = {
        "scenario_grid_promotion_token": 2,
        "alpha_shares_last_promotion_token": 2,
        "alpha_shares_last_promoted": ("Scenario Grid", 0.4, 0.3),
    }
    rerun = apply_promoted_alpha_shares(state, "Scenario Grid", 0.4, 0.3)
    assert rerun is False
    assert "portfolio_builder_autorun" not in state
