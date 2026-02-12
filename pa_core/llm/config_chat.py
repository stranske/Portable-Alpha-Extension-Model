"""Core config-chat apply helpers shared by dashboard wiring and tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, MutableMapping

from pa_core.llm.config_patch import (
    WIZARD_SESSION_MIRROR_KEYS,
    ConfigPatchValidationError,
    round_trip_validate_config,
    validate_patch_dict,
)
from pa_core.llm.config_patch import (
    apply_patch as apply_config_patch,
)
from pa_core.wizard.session_state import restore_wizard_session_snapshot
from pa_core.wizard_schema import AnalysisMode, DefaultConfigView

_APPLY_ACTION = "Apply"
_APPLY_VALIDATE_ACTION = "Apply+Validate"


def snapshot_wizard_session_state(
    config: DefaultConfigView,
    *,
    session_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture wizard config and all mirror key values for rollback/revert."""

    mirrors = sorted(set(WIZARD_SESSION_MIRROR_KEYS.values()))
    pre_mirror_values: dict[str, Any] = {}
    pre_missing_mirrors: list[str] = []
    for key in mirrors:
        if key in session_state:
            pre_mirror_values[key] = deepcopy(session_state[key])
        else:
            pre_missing_mirrors.append(key)
    return {
        "pre_apply_config": deepcopy(config),
        "pre_apply_mirrors": pre_mirror_values,
        "pre_apply_missing_mirrors": pre_missing_mirrors,
    }


def _mirror_session_state_value(key: str, value: Any) -> Any:
    if key == "analysis_mode" and isinstance(value, AnalysisMode):
        return value.value
    if key == "sleeve_constraint_scope" and value == "per_sleeve":
        return "sleeves"
    return value


def apply_patch_to_wizard_state(
    config: DefaultConfigView,
    patch: Any,
    *,
    session_state: MutableMapping[str, Any],
    wizard_config_key: str = "wizard_config",
) -> None:
    """Apply patch and synchronize wizard_config plus all mirror session keys."""

    apply_config_patch(config, patch, session_state=session_state)
    session_state[wizard_config_key] = config

    for config_key, mirror_key in WIZARD_SESSION_MIRROR_KEYS.items():
        session_state[mirror_key] = _mirror_session_state_value(
            config_key, getattr(config, config_key, None)
        )


def apply_config_chat_preview(
    preview: Mapping[str, Any],
    *,
    action: str,
    session_state: MutableMapping[str, Any],
    build_yaml_from_config: Callable[[Any], Mapping[str, Any]],
    wizard_config_key: str = "wizard_config",
    validate_config: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Apply preview patch with explicit Apply vs Apply+Validate behavior."""

    config = session_state.get(wizard_config_key)
    if not isinstance(config, DefaultConfigView):
        return False, f"{wizard_config_key} is not initialized."

    if action not in {_APPLY_ACTION, _APPLY_VALIDATE_ACTION}:
        return False, f"Unsupported action: {action}"

    raw_patch = preview.get("patch", {})
    try:
        patch = validate_patch_dict(raw_patch)
    except ConfigPatchValidationError as exc:
        if exc.unknown_paths:
            unknown_paths = ", ".join(exc.unknown_paths)
            return False, f"Patch validation failed: unknown patch fields at {unknown_paths}"
        return False, f"Patch validation failed: {exc}"
    except Exception as exc:
        return False, f"Patch validation failed: {exc}"

    if action == _APPLY_ACTION:
        apply_patch_to_wizard_state(
            config,
            patch,
            session_state=session_state,
            wizard_config_key=wizard_config_key,
        )
        return True, "Config changes applied."

    snapshot = snapshot_wizard_session_state(config, session_state=session_state)
    candidate = deepcopy(config)
    apply_config_patch(candidate, patch)
    validate_fn = validate_config or round_trip_validate_config
    validation_result = validate_fn(
        candidate,
        build_yaml_from_config=build_yaml_from_config,
    )
    if not validation_result.is_valid:
        restore_wizard_session_snapshot(
            snapshot,
            session_state=session_state,
            wizard_config_key=wizard_config_key,
        )
        errors = "; ".join(validation_result.errors)
        return False, f"Validation failed; changes not applied. {errors}"

    apply_patch_to_wizard_state(
        config,
        patch,
        session_state=session_state,
        wizard_config_key=wizard_config_key,
    )
    return True, "Config changes applied and validated."


__all__ = [
    "apply_config_chat_preview",
    "apply_patch_to_wizard_state",
    "snapshot_wizard_session_state",
]
