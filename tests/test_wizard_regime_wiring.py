import runpy

import streamlit as st

from pa_core.config import ModelConfig
from pa_core.wizard_schema import AnalysisMode, get_default_config


def _load_build_yaml() -> tuple[callable, dict]:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    return module["_build_yaml_from_config"], module


def test_build_yaml_includes_regime_switching_fields() -> None:
    st.session_state.clear()
    try:
        build_yaml, _module = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        config.regimes = [
            {"name": "calm", "idx_sigma_multiplier": 1.0},
            {"name": "stress", "idx_sigma_multiplier": 1.5, "sigma_H": 0.05},
        ]
        config.regime_transition = [[0.9, 0.1], [0.2, 0.8]]
        config.regime_start = "calm"

        yaml_dict = build_yaml(config)

        assert yaml_dict["regimes"] == config.regimes
        assert yaml_dict["regime_transition"] == config.regime_transition
        assert yaml_dict["regime_start"] == "calm"

        model_config = ModelConfig.model_validate(yaml_dict)
        assert model_config.regime_start == "calm"
        assert [regime.name for regime in model_config.regimes or []] == [
            "calm",
            "stress",
        ]
    finally:
        st.session_state.clear()
