import runpy

import pytest
import streamlit as st

from pa_core.config import ModelConfig
from pa_core.wizard_schema import AnalysisMode, get_default_config


def _load_build_yaml() -> callable:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    return module["_build_yaml_from_config"]


def _load_validate_regimes() -> callable:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    return module["_validate_regime_inputs"]


def test_wizard_regime_yaml_roundtrip() -> None:
    st.session_state.clear()
    try:
        build_yaml = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)

        config.regimes = [
            {"name": "Calm", "idx_sigma_multiplier": 0.8},
            {"name": "Stressed", "idx_sigma_multiplier": 1.3},
        ]
        config.regime_transition = [[0.9, 0.1], [0.2, 0.8]]
        config.regime_start = "Calm"

        yaml_dict = build_yaml(config)

        assert yaml_dict["regimes"] == config.regimes
        assert yaml_dict["regime_transition"] == config.regime_transition
        assert yaml_dict["regime_start"] == "Calm"

        model_config = ModelConfig.model_validate(yaml_dict)
        assert model_config.regimes is not None
        assert [regime.name for regime in model_config.regimes] == ["Calm", "Stressed"]
        assert model_config.regime_transition == [[0.9, 0.1], [0.2, 0.8]]
        assert model_config.regime_start == "Calm"
    finally:
        st.session_state.clear()


def test_wizard_regime_requires_transition() -> None:
    st.session_state.clear()
    try:
        build_yaml = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)
        config.regimes = [{"name": "Calm", "idx_sigma_multiplier": 0.8}]
        config.regime_transition = None

        with pytest.raises(
            ValueError, match="regime_transition is required when regimes are specified"
        ):
            build_yaml(config)
    finally:
        st.session_state.clear()


def test_wizard_regime_start_must_match_name() -> None:
    st.session_state.clear()
    try:
        build_yaml = _load_build_yaml()
        config = get_default_config(AnalysisMode.RETURNS)
        config.regimes = [
            {"name": "Calm", "idx_sigma_multiplier": 0.8},
            {"name": "Stressed", "idx_sigma_multiplier": 1.3},
        ]
        config.regime_transition = [[0.9, 0.1], [0.2, 0.8]]
        config.regime_start = "Unknown"

        with pytest.raises(ValueError, match="regime_start must match one of the regime names"):
            build_yaml(config)
    finally:
        st.session_state.clear()


def test_wizard_regime_validation_accepts_valid_inputs() -> None:
    validate_regimes = _load_validate_regimes()
    regimes = [
        {"name": "Calm", "idx_sigma_multiplier": 0.8},
        {"name": "Stressed", "idx_sigma_multiplier": 1.3},
    ]
    transition = [[0.9, 0.1], [0.2, 0.8]]

    validated_regimes, validated_transition, regime_names = validate_regimes(regimes, transition)

    assert validated_regimes == regimes
    assert validated_transition == transition
    assert regime_names == ["Calm", "Stressed"]


def test_wizard_regime_validation_accepts_mapping_inputs() -> None:
    validate_regimes = _load_validate_regimes()
    regimes = {
        "Calm": {"idx_sigma_multiplier": 0.8},
        "Stressed": {"idx_sigma_multiplier": 1.3},
    }
    transition = [[0.9, 0.1], [0.2, 0.8]]

    validated_regimes, validated_transition, regime_names = validate_regimes(regimes, transition)

    assert validated_regimes == [
        {"name": "Calm", "idx_sigma_multiplier": 0.8},
        {"name": "Stressed", "idx_sigma_multiplier": 1.3},
    ]
    assert validated_transition == transition
    assert regime_names == ["Calm", "Stressed"]


def test_wizard_regime_validation_accepts_tuple_transition() -> None:
    validate_regimes = _load_validate_regimes()
    regimes = [
        {"name": "Calm", "idx_sigma_multiplier": 0.8},
        {"name": "Stressed", "idx_sigma_multiplier": 1.3},
    ]
    transition = ((0.9, 0.1), (0.2, 0.8))

    validated_regimes, validated_transition, regime_names = validate_regimes(regimes, transition)

    assert validated_regimes == regimes
    assert validated_transition == [[0.9, 0.1], [0.2, 0.8]]
    assert regime_names == ["Calm", "Stressed"]


def test_wizard_regime_validation_rejects_non_square_transition() -> None:
    validate_regimes = _load_validate_regimes()
    regimes = [{"name": "Calm", "idx_sigma_multiplier": 0.8}]
    transition = [[1.0, 0.0]]

    with pytest.raises(
        ValueError, match="Transition matrix must be square and match the number of regimes"
    ):
        validate_regimes(regimes, transition)


def test_wizard_regime_validation_rejects_row_sum_error() -> None:
    validate_regimes = _load_validate_regimes()
    regimes = [{"name": "Calm", "idx_sigma_multiplier": 0.8}]
    transition = [[0.3]]

    with pytest.raises(ValueError, match="Transition matrix row 1 must sum to 1.0"):
        validate_regimes(regimes, transition)
