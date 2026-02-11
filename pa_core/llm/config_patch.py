"""Patch schema and validation helpers for wizard-config editing via LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from pa_core.backend import SUPPORTED_BACKENDS
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
