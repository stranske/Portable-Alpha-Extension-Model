from __future__ import annotations

import runpy
from copy import deepcopy

import streamlit as st

from pa_core.llm.config_patch import validate_patch_dict
from pa_core.llm.config_patch_chain import ConfigPatchChainResult
from pa_core.wizard_schema import AnalysisMode, get_default_config


def _load_module() -> dict[str, object]:
    return runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")


def _patch_preview_chain(module: dict[str, object], *, total_fund_capital: float = 1200.0) -> None:
    preview_fn = module["_preview_config_chat_change"]
    preview_fn.__globals__["_run_config_chat_instruction"] = lambda _instruction, config: (
        ConfigPatchChainResult(
            patch=validate_patch_dict(
                {
                    "set": {
                        "n_simulations": 5000,
                        "total_fund_capital": total_fund_capital,
                    },
                    "merge": {},
                    "remove": [],
                }
            ),
            summary="Increase simulations and update total capital.",
            risk_flags=[],
            unknown_output_keys=[],
            trace_url=None,
        )
    )


def test_preview_generation_does_not_mutate_live_wizard_config() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        _patch_preview_chain(module)

        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config
        before = deepcopy(module["_config_chat_snapshot"](config))

        module["_preview_config_chat_change"]("increase simulations to 5000")

        after = module["_config_chat_snapshot"](st.session_state["wizard_config"])
        assert before == after
    finally:
        st.session_state.clear()


def test_preview_generation_stores_unified_and_side_by_side_diff_keys() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        _patch_preview_chain(module)
        st.session_state["wizard_config"] = get_default_config(AnalysisMode.RETURNS)

        module["_preview_config_chat_change"]("increase simulations to 5000")

        assert isinstance(st.session_state.get("preview_unified_diff"), str)
        assert st.session_state["preview_unified_diff"].strip()
        assert isinstance(st.session_state.get("preview_sidebyside_diff"), str)
        assert st.session_state["preview_sidebyside_diff"].strip()
    finally:
        st.session_state.clear()


def test_apply_updates_wizard_config_and_session_mirrors_to_preview_state() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state["wizard_config"] = config
        st.session_state["wizard_total_fund_capital"] = config.total_fund_capital
        st.session_state[module["_CONFIG_CHAT_PREVIEW_KEY"]] = {
            "patch": {
                "set": {"n_simulations": 5000, "total_fund_capital": 900.0},
                "merge": {},
                "remove": [],
            }
        }

        ok, _message = module["_apply_preview_patch"](False)

        assert ok is True
        assert st.session_state["wizard_config"].n_simulations == 5000
        assert st.session_state["wizard_config"].total_fund_capital == 900.0
        assert st.session_state["wizard_total_fund_capital"] == 900.0
    finally:
        st.session_state.clear()


def test_revert_restores_previous_config_and_clears_preview_session_keys() -> None:
    st.session_state.clear()
    try:
        module = _load_module()
        _patch_preview_chain(module, total_fund_capital=900.0)

        config = get_default_config(AnalysisMode.RETURNS)
        config.total_fund_capital = 700.0
        st.session_state["wizard_config"] = config
        st.session_state["wizard_total_fund_capital"] = 700.0
        before = deepcopy(module["_config_chat_snapshot"](config))

        module["_preview_config_chat_change"]("increase simulations to 5000")
        ok_apply, _ = module["_apply_preview_patch"](False)
        assert ok_apply is True
        assert st.session_state["wizard_config"].total_fund_capital == 900.0

        ok_revert, _ = module["_revert_config_chat_change"]()

        assert ok_revert is True
        restored = module["_config_chat_snapshot"](st.session_state["wizard_config"])
        assert restored == before
        assert st.session_state["wizard_total_fund_capital"] == 700.0
        assert "preview_patch" not in st.session_state
        assert "preview_unified_diff" not in st.session_state
        assert "preview_sidebyside_diff" not in st.session_state
    finally:
        st.session_state.clear()
