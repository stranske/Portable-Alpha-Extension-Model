from __future__ import annotations

from typing import Any

from pa_core.llm.config_chat import apply_config_chat_preview
from pa_core.llm.config_patch import WIZARD_SESSION_MIRROR_KEYS
from pa_core.wizard_schema import AnalysisMode, get_default_config


def _expected_mirror_value(config_key: str, value: Any) -> Any:
    if config_key == "analysis_mode" and isinstance(value, AnalysisMode):
        return value.value
    if config_key == "sleeve_constraint_scope" and value == "per_sleeve":
        return "sleeves"
    return value


def _expected_mirror_state(config: Any) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for config_key, mirror_key in WIZARD_SESSION_MIRROR_KEYS.items():
        expected[mirror_key] = _expected_mirror_value(config_key, getattr(config, config_key, None))
    return expected


def test_apply_updates_config_and_all_mirrors_without_validation_call() -> None:
    config = get_default_config(AnalysisMode.RETURNS)
    session_state: dict[str, Any] = {"wizard_config": config}
    for mirror_key in sorted(set(WIZARD_SESSION_MIRROR_KEYS.values())):
        session_state[mirror_key] = "stale"

    def _must_not_validate(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("validation must not run for Apply")

    ok, message = apply_config_chat_preview(
        {
            "patch": {
                "set": {"n_simulations": 5000, "total_fund_capital": 1200.0},
                "merge": {},
                "remove": [],
            }
        },
        action="Apply",
        session_state=session_state,
        build_yaml_from_config=lambda _cfg: {},
        validate_config=_must_not_validate,
    )

    assert ok is True
    assert "applied" in message.lower()
    assert session_state["wizard_config"].n_simulations == 5000
    assert session_state["wizard_config"].total_fund_capital == 1200.0
    assert {
        key: session_state.get(key) for key in sorted(set(WIZARD_SESSION_MIRROR_KEYS.values()))
    } == _expected_mirror_state(session_state["wizard_config"])


def test_apply_persists_changes_even_if_validation_function_would_fail() -> None:
    config = get_default_config(AnalysisMode.RETURNS)
    session_state: dict[str, Any] = {"wizard_config": config}

    def _always_fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("should not be called for Apply")

    ok, _message = apply_config_chat_preview(
        {
            "patch": {
                "set": {"n_simulations": 5001},
                "merge": {},
                "remove": [],
            }
        },
        action="Apply",
        session_state=session_state,
        build_yaml_from_config=lambda _cfg: {},
        validate_config=_always_fail,
    )

    assert ok is True
    assert session_state["wizard_config"].n_simulations == 5001


def test_apply_validate_returns_errors_and_restores_snapshot_on_failure() -> None:
    config = get_default_config(AnalysisMode.RETURNS)
    session_state: dict[str, Any] = {"wizard_config": config}
    for mirror_key, expected in _expected_mirror_state(config).items():
        session_state[mirror_key] = expected
    before_config = session_state["wizard_config"]
    before_mirrors = {
        key: session_state[key] for key in sorted(set(WIZARD_SESSION_MIRROR_KEYS.values()))
    }
    validate_calls = {"count": 0}

    def _invalid(*_args: Any, **_kwargs: Any) -> Any:
        validate_calls["count"] += 1
        return type("Result", (), {"is_valid": False, "errors": ["boom"]})()

    ok, message = apply_config_chat_preview(
        {
            "patch": {
                "set": {"n_simulations": -1, "total_fund_capital": 1200.0},
                "merge": {},
                "remove": [],
            }
        },
        action="Apply+Validate",
        session_state=session_state,
        build_yaml_from_config=lambda _cfg: {},
        validate_config=_invalid,
    )

    assert ok is False
    assert "validation failed" in message.lower()
    assert validate_calls["count"] == 1
    assert session_state["wizard_config"] == before_config
    assert {
        key: session_state.get(key) for key in sorted(set(WIZARD_SESSION_MIRROR_KEYS.values()))
    } == before_mirrors


def test_apply_validate_applies_and_syncs_all_mirrors_on_success() -> None:
    config = get_default_config(AnalysisMode.RETURNS)
    session_state: dict[str, Any] = {"wizard_config": config}
    validate_calls = {"count": 0}

    def _valid(*_args: Any, **_kwargs: Any) -> Any:
        validate_calls["count"] += 1
        return type("Result", (), {"is_valid": True, "errors": []})()

    ok, message = apply_config_chat_preview(
        {
            "patch": {
                "set": {"n_simulations": 5000, "total_fund_capital": 1300.0},
                "merge": {},
                "remove": [],
            }
        },
        action="Apply+Validate",
        session_state=session_state,
        build_yaml_from_config=lambda _cfg: {},
        validate_config=_valid,
    )

    assert ok is True
    assert "validated" in message.lower()
    assert validate_calls["count"] == 1
    assert session_state["wizard_config"].n_simulations == 5000
    assert session_state["wizard_config"].total_fund_capital == 1300.0
    assert {
        key: session_state.get(key) for key in sorted(set(WIZARD_SESSION_MIRROR_KEYS.values()))
    } == _expected_mirror_state(session_state["wizard_config"])
