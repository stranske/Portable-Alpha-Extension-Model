"""Tests for wizard config patch allowlist and patch-format validation."""

from __future__ import annotations

import runpy

import pytest
import streamlit as st

from pa_core.llm.config_patch import (
    ALLOWED_WIZARD_PATCH_FIELDS,
    ConfigPatchValidationError,
    apply_patch,
    describe_allowed_patch_schema,
    diff_config,
    validate_patch,
    validate_round_trip,
)
from pa_core.wizard_schema import AnalysisMode, get_default_config


def test_allowed_schema_contains_expected_field_types_and_operations() -> None:
    schema = describe_allowed_patch_schema()

    assert "n_simulations" in schema
    assert schema["n_simulations"]["type"] == "int"
    assert schema["n_simulations"]["operations"] == ["set"]

    assert "risk_metrics" in schema
    assert "set" in schema["risk_metrics"]["operations"]
    assert "merge" in schema["risk_metrics"]["operations"]
    assert "enum" in schema["risk_metrics"]

    assert "regime_start" in schema
    assert "remove" in schema["regime_start"]["operations"]


def test_validate_patch_accepts_valid_set_merge_remove_payload() -> None:
    patch = {
        "set": {"n_simulations": 5000, "sleeve_validate_on_run": True},
        "merge": {"risk_metrics": ["Return", "Risk"]},
        "remove": ["regime_start", "regime_start"],
    }

    validated = validate_patch(patch)

    assert validated.set_ops["n_simulations"] == 5000
    assert validated.merge_ops["risk_metrics"] == ["Return", "Risk"]
    assert validated.remove_ops == ["regime_start"]


def test_validate_patch_rejects_unknown_root_operation() -> None:
    with pytest.raises(ConfigPatchValidationError, match="Unknown patch operation"):
        validate_patch({"replace": {"n_months": 24}})


def test_validate_patch_rejects_unknown_field() -> None:
    with pytest.raises(ConfigPatchValidationError, match="not allowlisted"):
        validate_patch({"set": {"totally_fake_field": 123}})


def test_validate_patch_rejects_type_mismatch() -> None:
    with pytest.raises(ConfigPatchValidationError, match="expects int"):
        validate_patch({"set": {"n_simulations": "5000"}})


def test_validate_patch_rejects_disallowed_merge_operation() -> None:
    with pytest.raises(ConfigPatchValidationError, match="does not support merge"):
        validate_patch({"merge": {"n_months": 36}})


def test_validate_patch_rejects_disallowed_remove_operation() -> None:
    with pytest.raises(ConfigPatchValidationError, match="does not support remove"):
        validate_patch({"remove": ["n_months"]})


def test_allowlist_is_explicit_and_non_empty() -> None:
    assert len(ALLOWED_WIZARD_PATCH_FIELDS) > 0
    assert "analysis_mode" in ALLOWED_WIZARD_PATCH_FIELDS
    assert "backend" in ALLOWED_WIZARD_PATCH_FIELDS


def test_apply_patch_updates_config_and_session_mirrors() -> None:
    cfg = get_default_config(AnalysisMode.RETURNS)
    session_state: dict[str, object] = {}

    apply_patch(
        cfg,
        {
            "set": {
                "n_simulations": 5000,
                "total_fund_capital": 900.0,
                "external_pa_capital": 200.0,
            },
            "merge": {"risk_metrics": ["terminal_ShortfallProb"]},
            "remove": ["regime_start"],
        },
        session_state=session_state,
    )

    assert cfg.n_simulations == 5000
    assert cfg.total_fund_capital == 900.0
    assert cfg.external_pa_capital == 200.0
    assert "terminal_ShortfallProb" in cfg.risk_metrics
    assert cfg.regime_start is None
    assert session_state["wizard_total_fund_capital"] == 900.0
    assert session_state["wizard_external_pa_capital"] == 200.0


def test_diff_config_returns_unified_diff() -> None:
    result = diff_config(
        {"N_SIMULATIONS": 1000, "N_MONTHS": 12},
        {"N_SIMULATIONS": 5000, "N_MONTHS": 12},
        format="json",
    )

    assert result.startswith("--- before\n+++ after\n")
    assert '"N_SIMULATIONS": 1000' in result
    assert '"N_SIMULATIONS": 5000' in result


def test_validate_round_trip_default_config_has_no_errors() -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        build_yaml_from_config = module["_build_yaml_from_config"]
        cfg = get_default_config(AnalysisMode.RETURNS)

        errors = validate_round_trip(cfg, build_yaml_from_config=build_yaml_from_config)

        assert errors == []
    finally:
        st.session_state.clear()
