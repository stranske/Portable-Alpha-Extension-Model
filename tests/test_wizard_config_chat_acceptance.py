from __future__ import annotations

import runpy

import streamlit as st

from pa_core.llm.config_patch import empty_patch, validate_patch_dict
from pa_core.llm.config_patch_chain import ConfigPatchChainResult
from pa_core.wizard_schema import AnalysisMode, get_default_config


def _load_module() -> dict[str, object]:
    return runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")


def test_preview_produces_diff_without_mutating_live_config() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        preview_fn = module["_preview_config_chat_instruction"]
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config

        preview_fn.__globals__["_run_config_chat_instruction"] = lambda _instruction, config: (
            ConfigPatchChainResult(
                patch=validate_patch_dict({"set": {"n_simulations": 5000}}),
                summary="Increase simulations",
                risk_flags=[],
                unknown_output_keys=[],
                trace_url=None,
            )
        )

        preview = preview_fn("increase simulations to 5000")

        assert st.session_state["wizard_config"].n_simulations == config.n_simulations
        assert preview["patch"]["set"]["n_simulations"] == 5000
        assert "-n_simulations: 1" in preview["unified_diff"]
        assert "+n_simulations: 5000" in preview["unified_diff"]
    finally:
        st.session_state.clear()


def test_apply_mutates_config_and_session_mirrors_without_validation_round_trip() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        apply_fn = module["_apply_config_chat_preview"]
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config
        st.session_state["wizard_total_fund_capital"] = config.total_fund_capital
        apply_fn.__globals__["round_trip_validate_config"] = lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AssertionError("round_trip_validate_config must not be called for Apply"))

        ok, message = apply_fn(
            {
                "patch": {
                    "set": {
                        "n_simulations": 5000,
                        "total_fund_capital": 1200.0,
                    },
                    "merge": {},
                    "remove": [],
                }
            },
            False,
        )

        assert ok is True
        assert "applied" in message.lower()
        assert st.session_state["wizard_config"].n_simulations == 5000
        assert st.session_state["wizard_total_fund_capital"] == 1200.0
    finally:
        st.session_state.clear()


def test_apply_validate_blocks_invalid_change_without_mutating_live_config() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        apply_fn = module["_apply_config_chat_preview"]
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config
        st.session_state["wizard_total_fund_capital"] = config.total_fund_capital
        original = config.n_simulations
        original_capital = config.total_fund_capital
        validate_calls: dict[str, int] = {"count": 0}

        def _fail_validation(*_args, **_kwargs):
            validate_calls["count"] += 1
            return type("Result", (), {"is_valid": False, "errors": ["boom"]})()

        apply_fn.__globals__["round_trip_validate_config"] = _fail_validation

        ok, message = apply_fn(
            {
                "patch": {
                    "set": {"n_simulations": -1, "total_fund_capital": 1200.0},
                    "merge": {},
                    "remove": [],
                }
            },
            True,
        )

        assert ok is False
        assert "validation failed" in message.lower()
        assert validate_calls["count"] == 1
        assert st.session_state["wizard_config"].n_simulations == original
        assert st.session_state["wizard_config"].total_fund_capital == original_capital
        assert st.session_state["wizard_total_fund_capital"] == original_capital
    finally:
        st.session_state.clear()


def test_apply_validate_applies_changes_and_mirrors_on_validation_success() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        apply_fn = module["_apply_config_chat_preview"]
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config
        st.session_state["wizard_total_fund_capital"] = config.total_fund_capital
        validate_calls: dict[str, int] = {"count": 0}

        def _ok_validation(*_args, **_kwargs):
            validate_calls["count"] += 1
            return type("Result", (), {"is_valid": True, "errors": []})()

        apply_fn.__globals__["round_trip_validate_config"] = _ok_validation

        ok, message = apply_fn(
            {
                "patch": {
                    "set": {
                        "n_simulations": 5000,
                        "total_fund_capital": 1200.0,
                    },
                    "merge": {},
                    "remove": [],
                }
            },
            True,
        )

        assert ok is True
        assert "validated" in message.lower()
        assert validate_calls["count"] == 1
        assert st.session_state["wizard_config"].n_simulations == 5000
        assert st.session_state["wizard_total_fund_capital"] == 1200.0
    finally:
        st.session_state.clear()


def test_preview_surfaces_risk_flag_for_rejected_unknown_output() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        preview_fn = module["_preview_config_chat_instruction"]
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config

        preview_fn.__globals__["_run_config_chat_instruction"] = lambda _instruction, config: (
            ConfigPatchChainResult(
                patch=empty_patch(),
                summary="Unknown keys rejected.",
                risk_flags=["rejected_unknown_patch_fields"],
                unknown_output_keys=[],
                trace_url=None,
            )
        )

        preview = preview_fn("change imaginary field")
        assert "rejected_unknown_patch_fields" in preview["risk_flags"]
    finally:
        st.session_state.clear()


def test_preview_carries_structured_unknown_output_keys() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        preview_fn = module["_preview_config_chat_instruction"]
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config

        preview_fn.__globals__["_run_config_chat_instruction"] = lambda _instruction, config: (
            ConfigPatchChainResult(
                patch=empty_patch(),
                summary="Unknown output fields were stripped.",
                risk_flags=[],
                unknown_output_keys=["hallucinated", "internal_meta"],
                trace_url=None,
            )
        )

        preview = preview_fn("change imaginary field")
        assert preview["unknown_output_keys"] == ["hallucinated", "internal_meta"]
        assert "stripped_unknown_output_keys" in preview["risk_flags"]
    finally:
        st.session_state.clear()


def test_unknown_patch_field_rejection_is_flagged() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        run_fn = module["_run_config_chat_instruction"]
        config = get_default_config(AnalysisMode.RETURNS)
        run_fn.__globals__["_build_config_chat_llm_invoke"] = lambda: (
            lambda _prompt: "{}",
            "openai",
            None,
        )
        run_fn.__globals__["run_config_patch_chain"] = lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("unknown wizard field: fake_field")
        )

        result = run_fn("set fake field to 1", config=config)
        assert "rejected_unknown_patch_fields" in result.risk_flags
    finally:
        st.session_state.clear()


def test_revert_restores_last_pre_apply_config_and_session_mirrors() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        apply_fn = module["_apply_config_chat_preview"]
        revert_fn = module["_revert_last_config_chat_change"]
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config
        st.session_state["wizard_total_fund_capital"] = 777.0

        ok, _ = apply_fn(
            {
                "patch": {
                    "set": {
                        "n_simulations": 5000,
                        "total_fund_capital": 1200.0,
                    },
                    "merge": {},
                    "remove": [],
                }
            },
            False,
        )
        assert ok is True
        assert st.session_state["wizard_config"].n_simulations == 5000
        assert st.session_state["wizard_total_fund_capital"] == 1200.0

        reverted, message = revert_fn()
        assert reverted is True
        assert "reverted" in message.lower()
        assert st.session_state["wizard_config"].n_simulations == 1
        assert st.session_state["wizard_total_fund_capital"] == 777.0
    finally:
        st.session_state.clear()
