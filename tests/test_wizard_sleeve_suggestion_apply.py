import runpy

import streamlit as st

from pa_core.wizard_schema import AnalysisMode, get_default_config


def test_build_yaml_uses_session_state_capital() -> None:
    st.session_state.clear()
    try:
        module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
        build_yaml = module["_build_yaml_from_config"]

        config = get_default_config(AnalysisMode.RETURNS)
        st.session_state[module["_TOTAL_CAPITAL_KEY"]] = 999.0
        st.session_state[module["_EXTERNAL_CAPITAL_KEY"]] = 111.0
        st.session_state[module["_ACTIVE_CAPITAL_KEY"]] = 222.0
        st.session_state[module["_INTERNAL_CAPITAL_KEY"]] = 666.0

        yaml_dict = build_yaml(config)

        assert yaml_dict["total_fund_capital"] == 999.0
        assert yaml_dict["external_pa_capital"] == 111.0
        assert yaml_dict["active_ext_capital"] == 222.0
        assert yaml_dict["internal_pa_capital"] == 666.0
    finally:
        st.session_state.clear()
