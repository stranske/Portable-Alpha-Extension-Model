"""Patch schema and validation for wizard config chat updates."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

import yaml

from pa_core.config import load_config

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

# Session-state mirrors used by the Scenario Wizard page.
WIZARD_SESSION_MIRROR_KEYS: dict[str, str] = {
    "total_fund_capital": "wizard_total_fund_capital",
    "external_pa_capital": "wizard_external_pa_capital",
    "active_ext_capital": "wizard_active_ext_capital",
    "internal_pa_capital": "wizard_internal_pa_capital",
    "w_beta_h": "wizard_w_beta_h",
    "theta_extpa": "wizard_theta_extpa",
    "active_share": "wizard_active_share",
    "regimes": "wizard_regimes_yaml",
    "regime_transition": "wizard_regime_transition_yaml",
    "regime_start": "wizard_regime_start",
    "sleeve_max_te": "sleeve_max_te",
    "sleeve_max_breach": "sleeve_max_breach",
    "sleeve_max_cvar": "sleeve_max_cvar",
    "sleeve_max_shortfall": "sleeve_max_shortfall",
    "sleeve_constraint_scope": "sleeve_constraint_scope",
    "sleeve_validate_on_run": "sleeve_validate_on_run",
}


@dataclass(frozen=True)
class ConfigRoundTripValidationResult:
    """Result of serializing and reloading wizard config."""

    is_valid: bool
    errors: list[str]
    yaml_dict: dict[str, Any] | None = None


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


def apply_patch(
    config: Any,
    patch: ConfigPatch | Mapping[str, Any],
    *,
    session_state: MutableMapping[str, Any] | None = None,
    session_mirror_keys: Mapping[str, str] | None = None,
) -> Any:
    """Apply a validated config patch to wizard config and optional session-state mirrors."""

    normalized_patch = patch if isinstance(patch, ConfigPatch) else validate_patch_dict(patch)
    mirrors = dict(session_mirror_keys or WIZARD_SESSION_MIRROR_KEYS)

    for key, value in normalized_patch.set.items():
        coerced_value = _coerce_runtime_value(key, value)
        setattr(config, key, coerced_value)
        if session_state is not None and key in mirrors:
            session_state[mirrors[key]] = _session_state_value(key, coerced_value)

    for key, patch_value in normalized_patch.merge.items():
        if key != "regimes":
            continue
        merged = _merge_regimes(getattr(config, key, None), patch_value)
        setattr(config, key, merged)
        if session_state is not None and key in mirrors:
            session_state[mirrors[key]] = merged

    for key in normalized_patch.remove:
        setattr(config, key, None)
        if session_state is not None and key in mirrors:
            session_state[mirrors[key]] = None

    return config


def diff_config(
    before_config: Any,
    after_config: Any,
    *,
    format: str = "yaml",
) -> tuple[str, str, str]:
    """Return unified diff and serialized before/after snapshots."""

    before_text = _serialize_config_snapshot(before_config, format=format)
    after_text = _serialize_config_snapshot(after_config, format=format)
    diff_text = "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )
    return diff_text, before_text, after_text


def round_trip_validate_config(
    config: Any,
    *,
    build_yaml_from_config: Callable[[Any], Mapping[str, Any]],
) -> ConfigRoundTripValidationResult:
    """Validate config by serializing wizard YAML and reloading via ``load_config``."""

    try:
        yaml_dict = dict(build_yaml_from_config(config))
        load_config(yaml_dict)
    except Exception as exc:
        return ConfigRoundTripValidationResult(is_valid=False, errors=[str(exc)], yaml_dict=None)
    return ConfigRoundTripValidationResult(is_valid=True, errors=[], yaml_dict=yaml_dict)


def _session_state_value(key: str, value: Any) -> Any:
    if key == "analysis_mode" and isinstance(value, AnalysisMode):
        return value.value
    if key == "sleeve_constraint_scope" and value == "per_sleeve":
        return "sleeves"
    return value


def _coerce_runtime_value(key: str, value: Any) -> Any:
    if key == "analysis_mode" and isinstance(value, str):
        return AnalysisMode(value)
    if key == "risk_metrics":
        normalized: list[str] = []
        for metric in value:
            if isinstance(metric, RiskMetric):
                normalized.append(metric.value)
            else:
                normalized.append(str(metric))
        return normalized
    if key == "sleeve_constraint_scope" and value == "sleeves":
        return "per_sleeve"
    return value


def _merge_regimes(
    existing: list[dict[str, Any]] | dict[str, dict[str, Any]] | None,
    patch_value: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    merged = _normalize_regimes_mapping(existing)
    for regime_name, regime_patch in patch_value.items():
        if not isinstance(regime_patch, Mapping):
            continue
        target = dict(merged.get(regime_name, {}))
        target.update(dict(regime_patch))
        if "name" not in target:
            target["name"] = regime_name
        merged[regime_name] = target
    return merged


def _normalize_regimes_mapping(
    regimes: list[dict[str, Any]] | dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if regimes is None:
        return {}
    if isinstance(regimes, list):
        normalized: dict[str, dict[str, Any]] = {}
        for regime in regimes:
            name = str(regime.get("name", "")).strip()
            if not name:
                continue
            normalized[name] = dict(regime)
        return normalized
    return {str(name): dict(regime) for name, regime in regimes.items()}


def _serialize_config_snapshot(config: Any, *, format: str) -> str:
    payload = _normalize_serializable(
        {key: getattr(config, key, None) for key in ALLOWED_WIZARD_FIELDS}
    )
    if format == "yaml":
        return yaml.safe_dump(payload, sort_keys=True)
    if format == "json":
        return json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"
    raise ValueError("format must be 'yaml' or 'json'")


def _normalize_serializable(value: Any) -> Any:
    if isinstance(value, (AnalysisMode, RiskMetric)):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalize_serializable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_serializable(child) for child in value]
    if isinstance(value, tuple):
        return [_normalize_serializable(child) for child in value]
    return value


__all__ = [
    "ALLOWED_WIZARD_FIELDS",
    "ConfigPatch",
    "ConfigPatchValidationError",
    "ConfigRoundTripValidationResult",
    "AllowedField",
    "WIZARD_SESSION_MIRROR_KEYS",
    "apply_patch",
    "allowed_wizard_schema",
    "diff_config",
    "empty_patch",
    "round_trip_validate_config",
    "validate_patch_dict",
]
