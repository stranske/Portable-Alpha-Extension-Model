"""Tests for wizard config patch schema/validation."""

from __future__ import annotations

import runpy
from copy import deepcopy

import pytest

from pa_core.llm.config_patch import (
    ConfigPatch,
    ConfigPatchValidationError,
    PatchSchemaValidationResult,
    ValidationError,
    allowed_wizard_schema,
    apply_patch,
    diff_config,
    empty_patch,
    generate_side_by_side_diff,
    generate_unified_diff,
    parse_chain_output,
    round_trip_validate_config,
    side_by_side_diff_config,
    validate_patch_dict,
    validate_patch_schema,
)
from pa_core.wizard_schema import AnalysisMode, get_default_config


def test_allowed_wizard_schema_contains_core_keys_and_type_labels() -> None:
    schema = allowed_wizard_schema()

    assert schema["n_simulations"] == "int"
    assert schema["analysis_mode"] == "AnalysisMode"
    assert schema["total_fund_capital"] == "float"


def test_validation_error_alias_points_to_config_patch_validation_error() -> None:
    assert ValidationError is ConfigPatchValidationError


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


def test_validate_patch_dict_reports_unknown_field_paths_structured() -> None:
    with pytest.raises(ConfigPatchValidationError) as exc_info:
        validate_patch_dict({"set": {"totally_fake_field": 1}})

    exc = exc_info.value
    assert exc.unknown_keys == ["totally_fake_field"]
    assert exc.unknown_paths == ["patch.set.totally_fake_field"]


def test_validate_patch_dict_reports_unknown_patch_ops_structured() -> None:
    with pytest.raises(ConfigPatchValidationError) as exc_info:
        validate_patch_dict({"set": {}, "unexpected_op": {"foo": 1}})

    exc = exc_info.value
    assert exc.unknown_keys == ["unexpected_op"]
    assert exc.unknown_paths == ["patch.unexpected_op"]


def test_validate_patch_dict_reports_non_string_unknown_patch_ops_structured() -> None:
    with pytest.raises(ConfigPatchValidationError) as exc_info:
        validate_patch_dict({"set": {}, 1: {"foo": 1}})

    exc = exc_info.value
    assert exc.unknown_keys == ["1"]
    assert exc.unknown_paths == ["patch.1"]


def test_validate_patch_schema_returns_structured_unknown_keys_result() -> None:
    result = validate_patch_schema({"set": {}, "nope": {}})

    assert isinstance(result, PatchSchemaValidationResult)
    assert result.is_valid is False
    assert result.unknown_keys == ["nope"]
    assert result.normalized_patch is None


def test_validate_patch_schema_normalizes_non_string_unknown_keys() -> None:
    result = validate_patch_schema({"set": {}, 5: {}, ("merge",): {}})

    assert isinstance(result, PatchSchemaValidationResult)
    assert result.is_valid is False
    assert result.unknown_keys == ["('merge',)", "5"]
    assert result.normalized_patch is None


def test_validate_patch_dict_type_error_exposes_field_type_metadata() -> None:
    with pytest.raises(ValidationError) as exc_info:
        validate_patch_dict({"set": {}, "remove": "not-a-list"})

    exc = exc_info.value
    assert exc.field_name == "remove"
    assert exc.expected_type == "list[str]"
    assert exc.actual_type == "str"


def test_validate_patch_dict_set_type_error_exposes_field_type_metadata() -> None:
    with pytest.raises(ConfigPatchValidationError) as exc_info:
        validate_patch_dict({"set": []})

    exc = exc_info.value
    assert exc.field_name == "set"
    assert exc.expected_type == "dict"
    assert exc.actual_type == "list"


def test_validate_patch_dict_merge_type_error_exposes_field_type_metadata() -> None:
    with pytest.raises(ConfigPatchValidationError) as exc_info:
        validate_patch_dict({"merge": []})

    exc = exc_info.value
    assert exc.field_name == "merge"
    assert exc.expected_type == "dict"
    assert exc.actual_type == "list"


def test_validate_patch_dict_merge_child_type_error_exposes_field_type_metadata() -> None:
    with pytest.raises(ConfigPatchValidationError) as exc_info:
        validate_patch_dict({"merge": {"regimes": []}})

    exc = exc_info.value
    assert exc.field_name == "merge.regimes"
    assert exc.expected_type == "dict"
    assert exc.actual_type == "list"


def test_validate_patch_dict_rejects_non_dict_top_level_patch_payload() -> None:
    with pytest.raises(ConfigPatchValidationError) as exc_info:
        validate_patch_dict(
            [
                {
                    "op": "set",
                    "key": "n_simulations",
                    "value": 5000,
                }
            ]
        )

    exc = exc_info.value
    assert exc.field_name == "patch"
    assert exc.expected_type == "dict"
    assert exc.actual_type == "list"


def test_validate_patch_schema_rejects_non_dict_top_level_patch_payload() -> None:
    with pytest.raises(ConfigPatchValidationError) as exc_info:
        validate_patch_schema(
            [
                {"op": "set", "key": "n_simulations", "value": 5000},
            ]
        )

    exc = exc_info.value
    assert exc.field_name == "patch"
    assert exc.expected_type == "dict"
    assert exc.actual_type == "list"


def test_parse_chain_output_reports_unknown_top_level_keys_structured() -> None:
    result = parse_chain_output(
        {
            "patch": {"set": {"n_simulations": 5000}},
            "summary": "Increase simulations.",
            "risk_flags": [],
            "hallucinated": True,
        }
    )

    assert result.patch.set == {"n_simulations": 5000}
    assert result.unknown_output_keys == ["hallucinated"]
    assert "stripped_unknown_output_keys" in result.risk_flags


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


def test_apply_patch_updates_config_and_session_state_mirrors() -> None:
    config = get_default_config(AnalysisMode.RETURNS)
    config_before = deepcopy(config)
    session_state: dict[str, object] = {}

    patch = validate_patch_dict(
        {
            "set": {
                "n_simulations": 5000,
                "total_fund_capital": 1200.0,
                "analysis_mode": "capital",
                "sleeve_constraint_scope": "sleeves",
            },
            "merge": {
                "regimes": {
                    "Calm": {"idx_sigma_multiplier": 0.8},
                }
            },
            "remove": ["sleeve_max_cvar"],
        }
    )

    updated = apply_patch(config, patch, session_state=session_state)

    assert id(updated) != id(config)
    assert config == config_before
    assert updated.n_simulations == 5000
    assert updated.total_fund_capital == 1200.0
    assert updated.analysis_mode == AnalysisMode.CAPITAL
    assert updated.sleeve_constraint_scope == "per_sleeve"
    assert updated.regimes == {"Calm": {"idx_sigma_multiplier": 0.8, "name": "Calm"}}
    assert updated.sleeve_max_cvar is None
    assert session_state["wizard_total_fund_capital"] == 1200.0
    assert session_state["sleeve_constraint_scope"] == "sleeves"
    assert session_state["wizard_regimes_yaml"] == {
        "Calm": {"idx_sigma_multiplier": 0.8, "name": "Calm"}
    }
    assert session_state["sleeve_max_cvar"] is None


def test_diff_config_returns_unified_yaml_diff_and_snapshots() -> None:
    before = get_default_config(AnalysisMode.RETURNS)
    after = get_default_config(AnalysisMode.RETURNS)
    after.n_simulations = 5000

    unified_diff, before_text, after_text = diff_config(before, after)

    assert unified_diff.startswith("--- before\n+++ after\n")
    assert "n_simulations: 1" in before_text
    assert "n_simulations: 5000" in after_text
    assert "-n_simulations: 1" in unified_diff
    assert "+n_simulations: 5000" in unified_diff


def test_side_by_side_diff_config_returns_structured_changed_rows() -> None:
    before = get_default_config(AnalysisMode.RETURNS)
    after = deepcopy(before)
    after.n_simulations = 5000

    rows = side_by_side_diff_config(before, after)

    assert rows
    assert any(row["changed"] for row in rows)
    assert any("n_simulations: 1" in row["before_line"] for row in rows)
    assert any("n_simulations: 5000" in row["after_line"] for row in rows)


def test_generate_unified_diff_returns_changed_plus_minus_lines() -> None:
    diff_text = generate_unified_diff("a: 1\n", "a: 2\n")

    assert "-a: 1" in diff_text
    assert "+a: 2" in diff_text


def test_generate_side_by_side_diff_returns_changed_line_pairs() -> None:
    rows = generate_side_by_side_diff("n_simulations: 1\n", "n_simulations: 5000\n")

    assert rows
    assert any(row["changed"] for row in rows)
    assert any("n_simulations: 1" in row["before_line"] for row in rows)
    assert any("n_simulations: 5000" in row["after_line"] for row in rows)


def test_round_trip_validate_config_success_and_error_paths() -> None:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    build_yaml_from_config = module["_build_yaml_from_config"]
    config = get_default_config(AnalysisMode.RETURNS)

    ok_result = round_trip_validate_config(config, build_yaml_from_config=build_yaml_from_config)
    assert ok_result.is_valid is True
    assert ok_result.errors == []
    assert isinstance(ok_result.yaml_dict, dict)

    config.n_simulations = -1
    invalid_result = round_trip_validate_config(
        config, build_yaml_from_config=build_yaml_from_config
    )
    assert invalid_result.is_valid is False
    assert invalid_result.errors
