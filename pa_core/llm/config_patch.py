"""Patch schema and validation for wizard config chat updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pa_core.wizard_schema import AnalysisMode, RiskMetric

_PATCH_KEYS: tuple[str, str, str] = ("set", "merge", "remove")


class ConfigPatchValidationError(ValueError):
    """Raised when an incoming config patch payload is invalid."""


@dataclass(frozen=True)
class AllowedField:
    """Schema metadata for a single wizard field."""

    type_label: str
    removable: bool = False
    mergeable: bool = False


@dataclass(frozen=True)
class ConfigPatch:
    """Validated patch structure for wizard config edits."""

    set: dict[str, Any]
    merge: dict[str, dict[str, Any]]
    remove: list[str]


# Explicit allowlist for wizard-facing fields. Keep this list tight and auditable.
ALLOWED_WIZARD_FIELDS: dict[str, AllowedField] = {
    "analysis_mode": AllowedField("AnalysisMode"),
    "n_simulations": AllowedField("int"),
    "n_months": AllowedField("int"),
    "financing_mode": AllowedField("str"),
    "external_pa_capital": AllowedField("float"),
    "active_ext_capital": AllowedField("float"),
    "internal_pa_capital": AllowedField("float"),
    "total_fund_capital": AllowedField("float"),
    "w_beta_h": AllowedField("float"),
    "w_alpha_h": AllowedField("float"),
    "theta_extpa": AllowedField("float"),
    "active_share": AllowedField("float"),
    "mu_h": AllowedField("float"),
    "mu_e": AllowedField("float"),
    "mu_m": AllowedField("float"),
    "sigma_h": AllowedField("float"),
    "sigma_e": AllowedField("float"),
    "sigma_m": AllowedField("float"),
    "rho_idx_h": AllowedField("float"),
    "rho_idx_e": AllowedField("float"),
    "rho_idx_m": AllowedField("float"),
    "rho_h_e": AllowedField("float"),
    "rho_h_m": AllowedField("float"),
    "rho_e_m": AllowedField("float"),
    "risk_metrics": AllowedField("list[str|RiskMetric]"),
    "sleeve_max_te": AllowedField("float|None", removable=True),
    "sleeve_max_breach": AllowedField("float|None", removable=True),
    "sleeve_max_cvar": AllowedField("float|None", removable=True),
    "sleeve_max_shortfall": AllowedField("float|None", removable=True),
    "sleeve_constraint_scope": AllowedField("'total'|'per_sleeve'"),
    "sleeve_validate_on_run": AllowedField("bool"),
    "return_distribution": AllowedField("'normal'|'student_t'"),
    "return_t_df": AllowedField("float"),
    "return_copula": AllowedField("'gaussian'|'t'"),
    "vol_regime": AllowedField("'single'|'two_state'"),
    "vol_regime_window": AllowedField("int"),
    "covariance_shrinkage": AllowedField("'none'|'ledoit_wolf'"),
    "correlation_repair_mode": AllowedField("'error'|'warn_fix'"),
    "correlation_repair_shrinkage": AllowedField("float"),
    "correlation_repair_max_abs_delta": AllowedField("float|None", removable=True),
    "backend": AllowedField("'numpy'"),
    "regimes": AllowedField("list[dict]|dict[str,dict]|None", removable=True, mergeable=True),
    "regime_transition": AllowedField("list[list[float]]|None", removable=True),
    "regime_start": AllowedField("str|None", removable=True),
}


def allowed_wizard_schema() -> dict[str, str]:
    """Return a serializable key->type schema for the config patch allowlist."""

    return {key: field.type_label for key, field in ALLOWED_WIZARD_FIELDS.items()}


def validate_patch_dict(raw_patch: Mapping[str, Any]) -> ConfigPatch:
    """Validate and normalize a config patch payload.

    Expected shape:
    ``{"set": {...}, "merge": {...}, "remove": [..]}``
    """

    if not isinstance(raw_patch, Mapping):
        raise ConfigPatchValidationError("patch must be a mapping")

    unknown_patch_ops = sorted(set(raw_patch) - set(_PATCH_KEYS))
    if unknown_patch_ops:
        raise ConfigPatchValidationError(
            f"unknown patch operations: {', '.join(unknown_patch_ops)}"
        )

    set_ops = _validate_set_ops(raw_patch.get("set", {}))
    merge_ops = _validate_merge_ops(raw_patch.get("merge", {}))
    remove_ops = _validate_remove_ops(raw_patch.get("remove", []))

    _ensure_no_duplicate_targets(set_ops=set_ops, merge_ops=merge_ops, remove_ops=remove_ops)

    return ConfigPatch(set=set_ops, merge=merge_ops, remove=remove_ops)


def _ensure_known_key(key: str) -> AllowedField:
    field = ALLOWED_WIZARD_FIELDS.get(key)
    if field is None:
        raise ConfigPatchValidationError(f"unknown wizard field: {key}")
    return field


def _validate_set_ops(raw_set: Any) -> dict[str, Any]:
    if not isinstance(raw_set, Mapping):
        raise ConfigPatchValidationError("patch.set must be a mapping")

    normalized: dict[str, Any] = {}
    for key, value in raw_set.items():
        if not isinstance(key, str):
            raise ConfigPatchValidationError("patch.set keys must be strings")
        field = _ensure_known_key(key)
        _validate_value_for_key(key, value)
        if value is None and not field.removable:
            raise ConfigPatchValidationError(f"field '{key}' does not support null values")
        normalized[key] = value
    return normalized


def _validate_merge_ops(raw_merge: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_merge, Mapping):
        raise ConfigPatchValidationError("patch.merge must be a mapping")

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in raw_merge.items():
        if not isinstance(key, str):
            raise ConfigPatchValidationError("patch.merge keys must be strings")
        field = _ensure_known_key(key)
        if not field.mergeable:
            raise ConfigPatchValidationError(f"field '{key}' does not support merge")
        if not isinstance(value, Mapping):
            raise ConfigPatchValidationError(f"patch.merge['{key}'] must be a mapping")
        normalized[key] = dict(value)
    return normalized


def _validate_remove_ops(raw_remove: Any) -> list[str]:
    if not isinstance(raw_remove, list):
        raise ConfigPatchValidationError("patch.remove must be a list")

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_remove:
        if not isinstance(value, str):
            raise ConfigPatchValidationError("patch.remove values must be strings")
        if value in seen:
            continue
        seen.add(value)
        field = _ensure_known_key(value)
        if not field.removable:
            raise ConfigPatchValidationError(f"field '{value}' does not support remove")
        normalized.append(value)
    return normalized


def _ensure_no_duplicate_targets(
    *,
    set_ops: Mapping[str, Any],
    merge_ops: Mapping[str, Mapping[str, Any]],
    remove_ops: list[str],
) -> None:
    set_keys = set(set_ops)
    merge_keys = set(merge_ops)
    remove_keys = set(remove_ops)

    duplicate_targets = sorted(
        (set_keys & merge_keys) | (set_keys & remove_keys) | (merge_keys & remove_keys)
    )
    if duplicate_targets:
        raise ConfigPatchValidationError(
            "patch cannot target the same field in multiple operations: "
            + ", ".join(duplicate_targets)
        )


def _validate_value_for_key(key: str, value: Any) -> None:
    if key == "analysis_mode":
        if isinstance(value, AnalysisMode):
            return
        if isinstance(value, str) and value in {m.value for m in AnalysisMode}:
            return
        raise ConfigPatchValidationError(
            "analysis_mode must be one of: capital, returns, alpha_shares, vol_mult"
        )

    if key in {
        "n_simulations",
        "n_months",
        "vol_regime_window",
    }:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigPatchValidationError(f"field '{key}' must be an int")
        return

    if key in {
        "sleeve_validate_on_run",
    }:
        if not isinstance(value, bool):
            raise ConfigPatchValidationError(f"field '{key}' must be a bool")
        return

    if key in {
        "financing_mode",
        "regime_start",
    }:
        if value is None and ALLOWED_WIZARD_FIELDS[key].removable:
            return
        if not isinstance(value, str):
            raise ConfigPatchValidationError(f"field '{key}' must be a string")
        return

    if key in {
        "backend",
    }:
        if value != "numpy":
            raise ConfigPatchValidationError("backend must be 'numpy'")
        return

    if key == "sleeve_constraint_scope":
        if value not in {"total", "per_sleeve", "sleeves"}:
            raise ConfigPatchValidationError(
                "sleeve_constraint_scope must be one of: total, per_sleeve, sleeves"
            )
        return

    if key == "return_distribution":
        if value not in {"normal", "student_t"}:
            raise ConfigPatchValidationError("return_distribution must be 'normal' or 'student_t'")
        return

    if key == "return_copula":
        if value not in {"gaussian", "t"}:
            raise ConfigPatchValidationError("return_copula must be 'gaussian' or 't'")
        return

    if key == "vol_regime":
        if value not in {"single", "two_state"}:
            raise ConfigPatchValidationError("vol_regime must be 'single' or 'two_state'")
        return

    if key == "covariance_shrinkage":
        if value not in {"none", "ledoit_wolf"}:
            raise ConfigPatchValidationError("covariance_shrinkage must be 'none' or 'ledoit_wolf'")
        return

    if key == "correlation_repair_mode":
        if value not in {"error", "warn_fix"}:
            raise ConfigPatchValidationError(
                "correlation_repair_mode must be 'error' or 'warn_fix'"
            )
        return

    if key == "risk_metrics":
        if not isinstance(value, list):
            raise ConfigPatchValidationError("risk_metrics must be a list")
        for metric in value:
            if isinstance(metric, RiskMetric):
                continue
            if isinstance(metric, str):
                continue
            raise ConfigPatchValidationError("risk_metrics entries must be strings or RiskMetric")
        return

    if key == "regimes":
        if value is None:
            return
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return
        if isinstance(value, dict) and all(
            isinstance(child_key, str) and isinstance(child_value, dict)
            for child_key, child_value in value.items()
        ):
            return
        raise ConfigPatchValidationError("regimes must be a list[dict], dict[str, dict], or null")

    if key == "regime_transition":
        if value is None:
            return
        if isinstance(value, list) and all(
            isinstance(row, list)
            and all(isinstance(cell, (int, float)) and not isinstance(cell, bool) for cell in row)
            for row in value
        ):
            return
        raise ConfigPatchValidationError(
            "regime_transition must be a list of numeric lists or null"
        )

    if key == "correlation_repair_max_abs_delta" and value is None:
        return

    if (
        key
        in {
            "sleeve_max_te",
            "sleeve_max_breach",
            "sleeve_max_cvar",
            "sleeve_max_shortfall",
        }
        and value is None
    ):
        return

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigPatchValidationError(f"field '{key}' must be a float")


def empty_patch() -> ConfigPatch:
    """Return an empty validated patch object."""

    return ConfigPatch(set={}, merge={}, remove=[])


__all__ = [
    "ALLOWED_WIZARD_FIELDS",
    "ConfigPatch",
    "ConfigPatchValidationError",
    "AllowedField",
    "allowed_wizard_schema",
    "empty_patch",
    "validate_patch_dict",
]
