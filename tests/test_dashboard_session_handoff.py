import runpy

import pandas as pd
import streamlit as st

from dashboard.utils import (
    CURRENT_INDEX_RETURNS_KEY,
    CURRENT_RESULTS_PATH_KEY,
    CURRENT_SCENARIO_CONFIG_KEY,
    config_capital_defaults,
    current_index_returns,
    current_results_path,
    current_scenario_config,
    remember_current_scenario,
)
from pa_core.config import ModelConfig


def _model_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "N_SIMULATIONS": 321,
            "N_MONTHS": 18,
            "total_fund_capital": 900.0,
            "external_pa_capital": 225.0,
            "active_ext_capital": 175.0,
            "internal_pa_capital": 500.0,
            "financing_mode": "broadcast",
            "theta_extpa": 0.35,
            "active_share": 0.65,
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        }
    )


def test_remember_current_scenario_round_trips_session_values() -> None:
    state = {}
    config = _model_config()
    index_returns = pd.Series([0.01, -0.02, 0.03])

    remember_current_scenario(
        state,
        config=config,
        results_path="runs/current/Outputs.xlsx",
        index_returns=index_returns,
    )

    assert state[CURRENT_SCENARIO_CONFIG_KEY] is config
    assert state[CURRENT_RESULTS_PATH_KEY] == "runs/current/Outputs.xlsx"
    assert current_scenario_config(state) is config
    assert current_results_path(state) == "runs/current/Outputs.xlsx"
    assert current_index_returns(state).tolist() == [0.01, -0.02, 0.03]

    state[CURRENT_INDEX_RETURNS_KEY].iloc[0] = 99.0
    assert index_returns.iloc[0] == 0.01


def test_capital_defaults_follow_current_config() -> None:
    defaults = config_capital_defaults(_model_config())

    assert defaults == {
        "total_fund_capital": 900.0,
        "external_pa_capital": 225.0,
        "active_ext_capital": 175.0,
        "internal_pa_capital": 500.0,
    }


def test_results_default_path_uses_wizard_session_path() -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/4_Results.py")
        st.session_state[CURRENT_RESULTS_PATH_KEY] = "/tmp/wizard-results.xlsx"

        assert module["_default_results_path"]() == "/tmp/wizard-results.xlsx"
    finally:
        st.session_state.clear()


def test_wizard_index_upload_bytes_uses_core_monthly_tr_loader() -> None:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")

    series = module["_read_index_csv_bytes"](
        b"Date,Other,Monthly_TR\n2026-02-28,5,-0.02\n2026-01-31,6,0.01\n"
    )

    assert series.tolist() == [0.01, -0.02]
