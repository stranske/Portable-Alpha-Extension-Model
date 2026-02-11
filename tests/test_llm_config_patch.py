"""Tests for wizard config patch allowlist and patch-format validation."""

from __future__ import annotations

import pytest

from pa_core.llm.config_patch import (
    ALLOWED_WIZARD_PATCH_FIELDS,
    ConfigPatchValidationError,
    describe_allowed_patch_schema,
    validate_patch,
)


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
