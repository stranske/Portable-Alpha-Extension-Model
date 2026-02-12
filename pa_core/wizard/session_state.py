"""Wizard session-state snapshot helpers."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from pa_core.llm.config_patch import WIZARD_SESSION_MIRROR_KEYS


def restore_wizard_session_snapshot(
    snapshot: Mapping[str, Any],
    *,
    session_state: MutableMapping[str, Any],
    wizard_config_key: str = "wizard_config",
) -> bool:
    """Restore wizard config and all mirror session keys from a saved snapshot."""

    if "pre_apply_config" not in snapshot:
        return False

    session_state[wizard_config_key] = snapshot["pre_apply_config"]
    mirror_values = snapshot.get("pre_apply_mirrors", {})
    missing_mirrors = snapshot.get("pre_apply_missing_mirrors", [])

    if not isinstance(mirror_values, Mapping):
        mirror_values = {}
    if not isinstance(missing_mirrors, list):
        missing_mirrors = []

    all_mirror_keys = sorted(set(WIZARD_SESSION_MIRROR_KEYS.values()))
    missing_lookup = {str(key) for key in missing_mirrors}
    for mirror_key in all_mirror_keys:
        if mirror_key in mirror_values:
            session_state[mirror_key] = mirror_values[mirror_key]
            continue
        if mirror_key in missing_lookup:
            session_state.pop(mirror_key, None)

    return True


__all__ = ["restore_wizard_session_snapshot"]
