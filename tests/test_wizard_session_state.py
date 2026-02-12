from __future__ import annotations

from pa_core.llm.config_patch import WIZARD_SESSION_MIRROR_KEYS
from pa_core.wizard.session_state import restore_wizard_session_snapshot
from pa_core.wizard_schema import AnalysisMode, get_default_config


def test_restore_wizard_session_snapshot_restores_all_mirror_keys() -> None:
    original_config = get_default_config(AnalysisMode.RETURNS)
    mirror_values = {
        mirror_key: f"snapshot-value-{idx}"
        for idx, mirror_key in enumerate(sorted(set(WIZARD_SESSION_MIRROR_KEYS.values())))
    }
    missing_key = sorted(set(WIZARD_SESSION_MIRROR_KEYS.values()))[-1]
    mirror_values.pop(missing_key, None)

    snapshot = {
        "pre_apply_config": original_config,
        "pre_apply_mirrors": mirror_values,
        "pre_apply_missing_mirrors": [missing_key],
    }

    session_state = {
        "wizard_config": get_default_config(AnalysisMode.CAPITAL),
        **{mirror_key: "mutated" for mirror_key in sorted(set(WIZARD_SESSION_MIRROR_KEYS.values()))},
    }

    restored = restore_wizard_session_snapshot(snapshot, session_state=session_state)
    assert restored is True
    assert session_state["wizard_config"] is original_config

    for mirror_key in sorted(set(WIZARD_SESSION_MIRROR_KEYS.values())):
        if mirror_key == missing_key:
            assert mirror_key not in session_state
        else:
            assert session_state[mirror_key] == mirror_values[mirror_key]


def test_restore_wizard_session_snapshot_requires_config_snapshot() -> None:
    session_state = {"wizard_config": object()}
    restored = restore_wizard_session_snapshot({}, session_state=session_state)
    assert restored is False
