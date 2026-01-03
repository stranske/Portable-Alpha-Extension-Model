import runpy
from pathlib import Path

import streamlit as st

from pa_core.wizard_schema import AnalysisMode, RiskMetric, get_default_config


def _load_build_yaml() -> tuple[callable, dict]:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    return module["_build_yaml_from_config"], module


def test_build_yaml_maps_all_fields() -> None:
    st.session_state.clear()
    try:
        build_yaml, module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        config.analysis_mode = AnalysisMode.CAPITAL
        config.n_simulations = 1234
        config.n_months = 17
        config.financing_mode = "broadcast"

        config.external_pa_capital = 10.0
        config.active_ext_capital = 20.0
        config.internal_pa_capital = 30.0
        config.total_fund_capital = 60.0

        st.session_state[module["_TOTAL_CAPITAL_KEY"]] = 99.0
        st.session_state[module["_EXTERNAL_CAPITAL_KEY"]] = 11.0
        st.session_state[module["_ACTIVE_CAPITAL_KEY"]] = 22.0
        st.session_state[module["_INTERNAL_CAPITAL_KEY"]] = 66.0

        config.w_beta_h = 0.11
        config.w_alpha_h = 0.89
        config.theta_extpa = 0.22
        config.active_share = 0.33

        config.mu_h = 0.01
        config.mu_e = 0.02
        config.mu_m = 0.03
        config.sigma_h = 0.04
        config.sigma_e = 0.05
        config.sigma_m = 0.06

        config.rho_idx_h = 0.1
        config.rho_idx_e = 0.2
        config.rho_idx_m = 0.3
        config.rho_h_e = 0.4
        config.rho_h_m = 0.5
        config.rho_e_m = 0.6

        config.risk_metrics = ["Return", "Risk", "terminal_ShortfallProb"]

        st.session_state["financing_settings"] = {
            "financing_model": "simple_proxy",
            "reference_sigma": 0.02,
            "volatility_multiple": 4.0,
            "term_months": 2.5,
            "schedule_path": "ignored.csv",
        }

        yaml_dict = build_yaml(config)

        assert yaml_dict["N_SIMULATIONS"] == 1234
        assert yaml_dict["N_MONTHS"] == 17
        assert yaml_dict["analysis_mode"] == "capital"
        assert yaml_dict["financing_mode"] == "broadcast"

        assert yaml_dict["total_fund_capital"] == 99.0
        assert yaml_dict["external_pa_capital"] == 11.0
        assert yaml_dict["active_ext_capital"] == 22.0
        assert yaml_dict["internal_pa_capital"] == 66.0

        assert yaml_dict["w_beta_H"] == 0.11
        assert yaml_dict["w_alpha_H"] == 0.89
        assert yaml_dict["theta_extpa"] == 0.22
        assert yaml_dict["active_share"] == 0.33

        assert yaml_dict["mu_H"] == 0.01
        assert yaml_dict["mu_E"] == 0.02
        assert yaml_dict["mu_M"] == 0.03
        assert yaml_dict["sigma_H"] == 0.04
        assert yaml_dict["sigma_E"] == 0.05
        assert yaml_dict["sigma_M"] == 0.06

        assert yaml_dict["rho_idx_H"] == 0.1
        assert yaml_dict["rho_idx_E"] == 0.2
        assert yaml_dict["rho_idx_M"] == 0.3
        assert yaml_dict["rho_H_E"] == 0.4
        assert yaml_dict["rho_H_M"] == 0.5
        assert yaml_dict["rho_E_M"] == 0.6

        assert yaml_dict["risk_metrics"] == ["Return", "Risk", "terminal_ShortfallProb"]
        assert yaml_dict["reference_sigma"] == 0.02
        assert yaml_dict["volatility_multiple"] == 4.0
        assert yaml_dict["financing_model"] == "simple_proxy"
        assert yaml_dict["financing_schedule_path"] is None
        assert yaml_dict["financing_term_months"] == 2.5
    finally:
        st.session_state.clear()


def test_build_yaml_includes_schedule_path() -> None:
    st.session_state.clear()
    try:
        build_yaml, _module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        st.session_state["financing_settings"] = {
            "financing_model": "schedule",
            "reference_sigma": 0.03,
            "volatility_multiple": 5.5,
            "term_months": 4.0,
            "schedule_path": Path("schedule.csv"),
        }

        yaml_dict = build_yaml(config)

        assert yaml_dict["reference_sigma"] == 0.03
        assert yaml_dict["volatility_multiple"] == 5.5
        assert yaml_dict["financing_model"] == "schedule"
        assert yaml_dict["financing_schedule_path"] == "schedule.csv"
        assert yaml_dict["financing_term_months"] == 4.0
    finally:
        st.session_state.clear()


def test_build_yaml_dict_alias_matches_from_config() -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        build_yaml = module["_build_yaml_from_config"]
        build_yaml_dict = module["_build_yaml_dict"]
        config = get_default_config(AnalysisMode.RETURNS)

        assert build_yaml_dict(config) == build_yaml(config)
    finally:
        st.session_state.clear()


def test_build_yaml_serializes_risk_metric_enums() -> None:
    st.session_state.clear()
    try:
        build_yaml, _module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        config.risk_metrics = [RiskMetric.RETURN, RiskMetric.RISK, RiskMetric.SHORTFALL_PROB]

        yaml_dict = build_yaml(config)

        assert yaml_dict["risk_metrics"] == ["Return", "Risk", "terminal_ShortfallProb"]
    finally:
        st.session_state.clear()
