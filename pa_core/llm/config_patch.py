"""Patch schema and validation helpers for wizard-config editing via LLM."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from difflib import unified_diff
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import yaml

from pa_core.backend import SUPPORTED_BACKENDS
from pa_core.config import load_config
from pa_core.wizard_schema import AnalysisMode, RiskMetric


class ConfigPatchValidationError(ValueError):
    """Raised when a config patch violates the allowed schema."""


@dataclass(frozen=True)
class WizardFieldSpec:
    """Validation metadata for one allowlisted wizard field."""

    type_name: str
    validator: Callable[[Any], bool]
    allow_set: bool = True
    allow_merge: bool = False
    allow_remove: bool = False
    enum_values: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ValidatedConfigPatch:
    """Normalized patch that passed schema and value validation."""

    set_ops: dict[str, Any]
    merge_ops: dict[str, Any]
    remove_ops: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "set": dict(self.set_ops),
            "merge": dict(self.merge_ops),
            "remove": list(self.remove_ops),
        }


_SESSION_FIELD_MIRRORS: dict[str, str] = {
    "total_fund_capital": "wizard_total_fund_capital",
    "external_pa_capital": "wizard_external_pa_capital",
    "active_ext_capital": "wizard_active_ext_capital",
    "internal_pa_capital": "wizard_internal_pa_capital",
    "w_beta_h": "wizard_w_beta_h",
    "theta_extpa": "wizard_theta_extpa",
    "active_share": "wizard_active_share",
    "sleeve_max_te": "sleeve_max_te",
    "sleeve_max_breach": "sleeve_max_breach",
    "sleeve_max_cvar": "sleeve_max_cvar",
    "sleeve_max_shortfall": "sleeve_max_shortfall",
    "sleeve_constraint_scope": "sleeve_constraint_scope",
    "sleeve_validate_on_run": "sleeve_validate_on_run",
    "regimes": "wizard_regimes_yaml",
    "regime_transition": "wizard_regime_transition_yaml",
    "regime_start": "wizard_regime_start",
}


def get_session_field_mirrors() -> dict[str, str]:
    """Return a copy of wizard-field -> session-state mirror key mappings."""

    return dict(_SESSION_FIELD_MIRRORS)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_string(value: Any) -> bool:
    return isinstance(value, str)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_regimes(value: Any) -> bool:
    if isinstance(value, list):
        return all(isinstance(item, dict) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and isinstance(item, Mapping) for key, item in value.items()
        )
    return False


def _is_transition_matrix(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    rows = list(value)
    if not rows:
        return False
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return False
        if not all(_is_number(cell) for cell in row):
            return False
    return True


_ANALYSIS_MODE_VALUES = tuple(mode.value for mode in AnalysisMode)
_RISK_METRIC_VALUES = tuple(metric.value for metric in RiskMetric)
_SCOPE_VALUES = ("total", "per_sleeve")
_RETURN_DISTRIBUTION_VALUES = ("normal", "student_t")
_RETURN_COPULA_VALUES = ("gaussian", "t")
_VOL_REGIME_VALUES = ("single", "two_state")
_COVARIANCE_SHRINKAGE_VALUES = ("none", "ledoit_wolf")
_CORRELATION_REPAIR_MODE_VALUES = ("error", "warn_fix")


def _enum_validator(allowed: tuple[str, ...]) -> Callable[[Any], bool]:
    return lambda value: isinstance(value, str) and value in allowed


ALLOWED_WIZARD_PATCH_FIELDS: dict[str, WizardFieldSpec] = {
    "analysis_mode": WizardFieldSpec(
        type_name="enum",
        validator=_enum_validator(_ANALYSIS_MODE_VALUES),
        enum_values=_ANALYSIS_MODE_VALUES,
    ),
    "n_simulations": WizardFieldSpec(type_name="int", validator=_is_int),
    "n_months": WizardFieldSpec(type_name="int", validator=_is_int),
    "financing_mode": WizardFieldSpec(type_name="str", validator=_is_string),
    "external_pa_capital": WizardFieldSpec(type_name="float", validator=_is_number),
    "active_ext_capital": WizardFieldSpec(type_name="float", validator=_is_number),
    "internal_pa_capital": WizardFieldSpec(type_name="float", validator=_is_number),
    "total_fund_capital": WizardFieldSpec(type_name="float", validator=_is_number),
    "w_beta_h": WizardFieldSpec(type_name="float", validator=_is_number),
    "w_alpha_h": WizardFieldSpec(type_name="float", validator=_is_number),
    "theta_extpa": WizardFieldSpec(type_name="float", validator=_is_number),
    "active_share": WizardFieldSpec(type_name="float", validator=_is_number),
    "mu_h": WizardFieldSpec(type_name="float", validator=_is_number),
    "mu_e": WizardFieldSpec(type_name="float", validator=_is_number),
    "mu_m": WizardFieldSpec(type_name="float", validator=_is_number),
    "sigma_h": WizardFieldSpec(type_name="float", validator=_is_number),
    "sigma_e": WizardFieldSpec(type_name="float", validator=_is_number),
    "sigma_m": WizardFieldSpec(type_name="float", validator=_is_number),
    "rho_idx_h": WizardFieldSpec(type_name="float", validator=_is_number),
    "rho_idx_e": WizardFieldSpec(type_name="float", validator=_is_number),
    "rho_idx_m": WizardFieldSpec(type_name="float", validator=_is_number),
    "rho_h_e": WizardFieldSpec(type_name="float", validator=_is_number),
    "rho_h_m": WizardFieldSpec(type_name="float", validator=_is_number),
    "rho_e_m": WizardFieldSpec(type_name="float", validator=_is_number),
    "risk_metrics": WizardFieldSpec(
        type_name="list[str]",
        validator=_is_string_list,
        allow_merge=True,
        enum_values=_RISK_METRIC_VALUES,
    ),
    "sleeve_max_te": WizardFieldSpec(
        type_name="optional[float]", validator=_is_number, allow_remove=True
    ),
    "sleeve_max_breach": WizardFieldSpec(
        type_name="optional[float]", validator=_is_number, allow_remove=True
    ),
    "sleeve_max_cvar": WizardFieldSpec(
        type_name="optional[float]", validator=_is_number, allow_remove=True
    ),
    "sleeve_max_shortfall": WizardFieldSpec(
        type_name="optional[float]", validator=_is_number, allow_remove=True
    ),
    "sleeve_constraint_scope": WizardFieldSpec(
        type_name="enum",
        validator=_enum_validator(_SCOPE_VALUES),
        enum_values=_SCOPE_VALUES,
    ),
    "sleeve_validate_on_run": WizardFieldSpec(type_name="bool", validator=_is_bool),
    "return_distribution": WizardFieldSpec(
        type_name="enum",
        validator=_enum_validator(_RETURN_DISTRIBUTION_VALUES),
        enum_values=_RETURN_DISTRIBUTION_VALUES,
    ),
    "return_t_df": WizardFieldSpec(type_name="float", validator=_is_number),
    "return_copula": WizardFieldSpec(
        type_name="enum",
        validator=_enum_validator(_RETURN_COPULA_VALUES),
        enum_values=_RETURN_COPULA_VALUES,
    ),
    "vol_regime": WizardFieldSpec(
        type_name="enum",
        validator=_enum_validator(_VOL_REGIME_VALUES),
        enum_values=_VOL_REGIME_VALUES,
    ),
    "vol_regime_window": WizardFieldSpec(type_name="int", validator=_is_int),
    "covariance_shrinkage": WizardFieldSpec(
        type_name="enum",
        validator=_enum_validator(_COVARIANCE_SHRINKAGE_VALUES),
        enum_values=_COVARIANCE_SHRINKAGE_VALUES,
    ),
    "correlation_repair_mode": WizardFieldSpec(
        type_name="enum",
        validator=_enum_validator(_CORRELATION_REPAIR_MODE_VALUES),
        enum_values=_CORRELATION_REPAIR_MODE_VALUES,
    ),
    "correlation_repair_shrinkage": WizardFieldSpec(type_name="float", validator=_is_number),
    "correlation_repair_max_abs_delta": WizardFieldSpec(
        type_name="optional[float]",
        validator=_is_number,
        allow_remove=True,
    ),
    "backend": WizardFieldSpec(
        type_name="enum",
        validator=_enum_validator(tuple(SUPPORTED_BACKENDS)),
        enum_values=tuple(SUPPORTED_BACKENDS),
    ),
    "regimes": WizardFieldSpec(
        type_name="optional[list[dict] | dict[str, dict]]",
        validator=_is_regimes,
        allow_merge=True,
        allow_remove=True,
    ),
    "regime_transition": WizardFieldSpec(
        type_name="optional[list[list[float]]]",
        validator=_is_transition_matrix,
        allow_remove=True,
    ),
    "regime_start": WizardFieldSpec(
        type_name="optional[str]", validator=_is_string, allow_remove=True
    ),
}

_PATCH_ROOT_KEYS = frozenset({"set", "merge", "remove"})


def describe_allowed_patch_schema() -> dict[str, dict[str, Any]]:
    """Return a serializable schema view of allowlisted patchable wizard fields."""

    schema: dict[str, dict[str, Any]] = {}
    for key in sorted(ALLOWED_WIZARD_PATCH_FIELDS):
        spec = ALLOWED_WIZARD_PATCH_FIELDS[key]
        schema[key] = {
            "type": spec.type_name,
            "operations": [
                op
                for op, allowed in (
                    ("set", spec.allow_set),
                    ("merge", spec.allow_merge),
                    ("remove", spec.allow_remove),
                )
                if allowed
            ],
        }
        if spec.enum_values:
            schema[key]["enum"] = list(spec.enum_values)
    return schema


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigPatchValidationError(f"'{name}' must be a mapping of field -> value.")
    return {str(key): child for key, child in value.items()}


def _require_remove_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigPatchValidationError("'remove' must be a list of field names.")
    if not all(isinstance(item, str) for item in value):
        raise ConfigPatchValidationError("'remove' entries must all be strings.")
    return value


def _check_field_allowed(field: str, op: str) -> WizardFieldSpec:
    spec = ALLOWED_WIZARD_PATCH_FIELDS.get(field)
    if spec is None:
        raise ConfigPatchValidationError(f"Field '{field}' is not allowlisted for patching.")
    if op == "set" and not spec.allow_set:
        raise ConfigPatchValidationError(f"Field '{field}' does not support set operations.")
    if op == "merge" and not spec.allow_merge:
        raise ConfigPatchValidationError(f"Field '{field}' does not support merge operations.")
    if op == "remove" and not spec.allow_remove:
        raise ConfigPatchValidationError(f"Field '{field}' does not support remove operations.")
    return spec


def _validate_value(field: str, value: Any, spec: WizardFieldSpec) -> None:
    if not spec.validator(value):
        raise ConfigPatchValidationError(
            f"Field '{field}' expects {spec.type_name}, got {type(value).__name__}."
        )


def validate_patch(patch: Mapping[str, Any]) -> ValidatedConfigPatch:
    """Validate and normalize a patch with `set`/`merge`/`remove` operations."""

    if not isinstance(patch, Mapping):
        raise ConfigPatchValidationError("Patch payload must be a mapping.")

    unknown_root_keys = set(patch) - _PATCH_ROOT_KEYS
    if unknown_root_keys:
        keys = ", ".join(sorted(str(key) for key in unknown_root_keys))
        raise ConfigPatchValidationError(f"Unknown patch operation(s): {keys}.")

    set_ops = _require_mapping("set", patch.get("set"))
    merge_ops = _require_mapping("merge", patch.get("merge"))
    remove_ops = _require_remove_list(patch.get("remove"))

    for field, value in set_ops.items():
        spec = _check_field_allowed(field, "set")
        _validate_value(field, value, spec)

    for field, value in merge_ops.items():
        spec = _check_field_allowed(field, "merge")
        _validate_value(field, value, spec)

    deduped_remove: list[str] = []
    seen: set[str] = set()
    for field in remove_ops:
        _check_field_allowed(field, "remove")
        if field in seen:
            continue
        seen.add(field)
        deduped_remove.append(field)

    return ValidatedConfigPatch(set_ops=set_ops, merge_ops=merge_ops, remove_ops=deduped_remove)


def _merge_value(current: Any, incoming: Any) -> Any:
    if isinstance(current, Mapping) and isinstance(incoming, Mapping):
        merged = dict(current)
        merged.update(dict(incoming))
        return merged
    if isinstance(current, list) and isinstance(incoming, list):
        merged = list(current)
        for item in incoming:
            if item not in merged:
                merged.append(item)
        return merged
    return incoming


def _mirror_session_field(
    field: str,
    value: Any,
    *,
    session_state: MutableMapping[str, Any] | None,
) -> None:
    if session_state is None:
        return
    key = _SESSION_FIELD_MIRRORS.get(field)
    if key is None:
        return
    session_state[key] = deepcopy(value)


def _clear_session_field(field: str, *, session_state: MutableMapping[str, Any] | None) -> None:
    if session_state is None:
        return
    key = _SESSION_FIELD_MIRRORS.get(field)
    if key is None:
        return
    session_state.pop(key, None)


def apply_patch(
    config: Any,
    patch: Mapping[str, Any] | ValidatedConfigPatch,
    *,
    session_state: MutableMapping[str, Any] | None = None,
) -> Any:
    """Apply a validated patch to a wizard config and optional session-state mirrors."""

    validated = patch if isinstance(patch, ValidatedConfigPatch) else validate_patch(patch)

    for field, value in validated.set_ops.items():
        setattr(config, field, deepcopy(value))
        _mirror_session_field(field, value, session_state=session_state)

    for field, value in validated.merge_ops.items():
        current = getattr(config, field, None)
        merged = _merge_value(current, value)
        setattr(config, field, deepcopy(merged))
        _mirror_session_field(field, merged, session_state=session_state)

    for field in validated.remove_ops:
        setattr(config, field, None)
        _clear_session_field(field, session_state=session_state)

    return config


def _serialize_snapshot(snapshot: Mapping[str, Any], *, format: str) -> str:
    kind = format.strip().lower()
    if kind == "yaml":
        return yaml.safe_dump(dict(snapshot), sort_keys=True)
    if kind == "json":
        return json.dumps(dict(snapshot), indent=2, sort_keys=True, default=str) + "\n"
    raise ValueError("format must be 'yaml' or 'json'")


def diff_config(
    before_snapshot: Mapping[str, Any],
    after_snapshot: Mapping[str, Any],
    *,
    format: str = "yaml",
    fromfile: str = "before",
    tofile: str = "after",
) -> str:
    """Produce a unified diff between two config snapshots."""

    before_text = _serialize_snapshot(before_snapshot, format=format)
    after_text = _serialize_snapshot(after_snapshot, format=format)
    lines = unified_diff(
        before_text.splitlines(keepends=True),
        after_text.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
    )
    return "".join(lines)


def validate_round_trip(
    config: Any,
    *,
    build_yaml_from_config: Callable[[Any], Mapping[str, Any]],
) -> list[str]:
    """Validate config by serializing and reloading through ``load_config``."""

    try:
        yaml_payload = build_yaml_from_config(config)
    except Exception as exc:
        return [f"build_yaml_from_config failed: {exc}"]

    try:
        load_config(dict(yaml_payload))
    except Exception as exc:
        return [str(exc)]
    return []


def apply_patch_with_validation(
    config: Any,
    patch: Mapping[str, Any] | ValidatedConfigPatch,
    *,
    build_yaml_from_config: Callable[[Any], Mapping[str, Any]],
    session_state: MutableMapping[str, Any] | None = None,
) -> tuple[Any, list[str]]:
    """Apply patch then return round-trip validation errors, if any."""

    apply_patch(config, patch, session_state=session_state)
    errors = validate_round_trip(config, build_yaml_from_config=build_yaml_from_config)
    return config, errors
