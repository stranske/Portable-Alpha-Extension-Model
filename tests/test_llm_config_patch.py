"""Tests for wizard config patch schema/validation."""

from __future__ import annotations

import pytest

from pa_core.llm.config_patch import (
    ConfigPatch,
    ConfigPatchValidationError,
    allowed_wizard_schema,
    empty_patch,
    validate_patch_dict,
)


def test_allowed_wizard_schema_contains_core_keys_and_type_labels() -> None:
    schema = allowed_wizard_schema()

    assert schema["n_simulations"] == "int"
    assert schema["analysis_mode"] == "AnalysisMode"
    assert schema["total_fund_capital"] == "float"


def test_validate_patch_dict_accepts_valid_set_payload() -> None:
    patch = validate_patch_dict(
        {
            "set": {
                "analysis_mode": "returns",
                "n_simulations": 5000,
                "sleeve_max_breach": 0.3,
            }
        }
    )

    assert isinstance(patch, ConfigPatch)
    assert patch.set["n_simulations"] == 5000
    assert patch.merge == {}
    assert patch.remove == []


def test_validate_patch_dict_rejects_unknown_field() -> None:
    with pytest.raises(ConfigPatchValidationError, match="unknown wizard field"):
        validate_patch_dict({"set": {"totally_fake_field": 1}})


def test_validate_patch_dict_rejects_wrong_type() -> None:
    with pytest.raises(ConfigPatchValidationError, match="must be an int"):
        validate_patch_dict({"set": {"n_simulations": "5000"}})


def test_validate_patch_dict_allows_merge_for_regimes_only() -> None:
    patch = validate_patch_dict({"merge": {"regimes": {"stress": {"mu_H": 0.02}}}})
    assert patch.merge["regimes"]["stress"]["mu_H"] == 0.02

    with pytest.raises(ConfigPatchValidationError, match="does not support merge"):
        validate_patch_dict({"merge": {"n_simulations": {"value": 5000}}})


def test_validate_patch_dict_restricts_remove_to_removable_keys() -> None:
    patch = validate_patch_dict({"remove": ["regime_start", "sleeve_max_cvar"]})
    assert patch.remove == ["regime_start", "sleeve_max_cvar"]

    with pytest.raises(ConfigPatchValidationError, match="does not support remove"):
        validate_patch_dict({"remove": ["n_simulations"]})


def test_validate_patch_dict_rejects_duplicate_targets_across_operations() -> None:
    with pytest.raises(ConfigPatchValidationError, match="multiple operations"):
        validate_patch_dict({"set": {"regime_start": "base"}, "remove": ["regime_start"]})


def test_empty_patch_returns_valid_empty_container() -> None:
    assert empty_patch() == ConfigPatch(set={}, merge={}, remove=[])
