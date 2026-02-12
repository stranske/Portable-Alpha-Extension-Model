from __future__ import annotations

import runpy

import streamlit as st

from pa_core.wizard_schema import AnalysisMode, get_default_config


def _mute_streamlit_ui(module: dict, monkeypatch) -> None:
    for name in ("success", "warning", "error", "caption", "markdown", "code", "info"):
        monkeypatch.setattr(module["st"], name, lambda *args, **kwargs: None)


def test_preview_generates_diff_without_mutating_live_config(monkeypatch) -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        _mute_streamlit_ui(module, monkeypatch)

        config = get_default_config(AnalysisMode.RETURNS)
        config.n_simulations = 1000
        config.sleeve_max_breach = 0.2
        st.session_state["wizard_config"] = config

        module["_preview_config_chat_change"](
            "increase simulations to 5000 and reduce breach tolerance"
        )

        preview = st.session_state.get(module["_CONFIG_CHAT_PREVIEW_KEY"])
        assert isinstance(preview, dict)
        assert st.session_state["wizard_config"].n_simulations == 1000
        assert "n_simulations" in preview["unified_diff"]
        assert preview["patch"]["set"]["n_simulations"] == 5000
    finally:
        st.session_state.clear()


def test_apply_validate_blocks_invalid_patch_without_mutation(monkeypatch) -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        _mute_streamlit_ui(module, monkeypatch)

        config = get_default_config(AnalysisMode.RETURNS)
        config.n_months = 12
        st.session_state["wizard_config"] = config
        st.session_state[module["_CONFIG_CHAT_PREVIEW_KEY"]] = {
            "patch": {"set": {"n_months": 0}, "merge": {}, "remove": []},
            "instruction": "set months to 0",
        }

        module["_apply_preview_patch"](validate_first=True)

        assert st.session_state["wizard_config"].n_months == 12
        assert st.session_state.get(module["_CONFIG_CHAT_HISTORY_KEY"], []) == []
    finally:
        st.session_state.clear()


def test_revert_restores_previous_config_and_session_mirrors(monkeypatch) -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        _mute_streamlit_ui(module, monkeypatch)

        config = get_default_config(AnalysisMode.RETURNS)
        config.total_fund_capital = 700.0
        st.session_state["wizard_config"] = config
        st.session_state["wizard_total_fund_capital"] = 700.0
        st.session_state[module["_CONFIG_CHAT_PREVIEW_KEY"]] = {
            "patch": {"set": {"total_fund_capital": 900.0}, "merge": {}, "remove": []},
            "instruction": "set total capital to 900",
        }

        module["_apply_preview_patch"](validate_first=False)
        assert st.session_state["wizard_config"].total_fund_capital == 900.0
        assert st.session_state["wizard_total_fund_capital"] == 900.0

        module["_revert_config_chat_change"]()
        assert st.session_state["wizard_config"].total_fund_capital == 700.0
        assert st.session_state["wizard_total_fund_capital"] == 700.0
    finally:
        st.session_state.clear()
