import runpy
from pathlib import Path

import pytest
import streamlit as st

from pa_core.config import RegimeConfig
from pa_core.wizard_schema import AnalysisMode, RiskMetric, get_default_config


def _load_helpers() -> dict:
    return runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")


def test_normalize_risk_metric_defaults_handles_mixed_inputs() -> None:
    helpers = _load_helpers()
    normalize = helpers["_normalize_risk_metric_defaults"]

    result = normalize(["Return", RiskMetric.RISK, "Unknown"])

    assert result == [RiskMetric.RETURN, RiskMetric.RISK]


def test_normalize_risk_metric_defaults_falls_back_to_all_metrics() -> None:
    helpers = _load_helpers()
    normalize = helpers["_normalize_risk_metric_defaults"]

    result = normalize([])

    assert result == list(RiskMetric)


def test_serialize_risk_metrics_handles_mixed_inputs() -> None:
    helpers = _load_helpers()
    serialize = helpers["_serialize_risk_metrics"]

    result = serialize([RiskMetric.RETURN, "Risk", 123])

    assert result == ["Return", "Risk", "123"]


def test_parse_yaml_or_json_accepts_valid_and_rejects_invalid() -> None:
    helpers = _load_helpers()
    parse = helpers["_parse_yaml_or_json"]

    parsed = parse("a: 1\nb:\n  - 2\n  - 3", "Payload")

    assert parsed == {"a": 1, "b": [2, 3]}

    with pytest.raises(ValueError, match="Payload must be valid YAML/JSON"):
        parse("a: [1, 2", "Payload")


def test_serialize_regimes_accepts_models_and_dicts() -> None:
    helpers = _load_helpers()
    serialize_regimes = helpers["_serialize_regimes"]

    regimes = [
        RegimeConfig(name="Calm", idx_sigma_multiplier=0.8),
        {"name": "Stress", "idx_sigma_multiplier": 1.2},
    ]

    serialized = serialize_regimes(regimes)

    assert serialized[0]["name"] == "Calm"
    assert serialized[0]["idx_sigma_multiplier"] == 0.8
    assert serialized[1] == {"name": "Stress", "idx_sigma_multiplier": 1.2}

    with pytest.raises(ValueError, match="Regime entries must be dicts or model instances"):
        serialize_regimes([object()])


def test_normalize_sleeve_constraint_scope_maps_aliases() -> None:
    helpers = _load_helpers()
    normalize = helpers["_normalize_sleeve_constraint_scope"]

    assert normalize("sleeves") == "per_sleeve"
    assert normalize("per_sleeve") == "per_sleeve"
    assert normalize("total") == "total"
    assert normalize(None) == "total"


def test_validate_regime_inputs_accepts_mapping_and_transition() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = {
        "Calm": {"idx_sigma_multiplier": 0.8},
        "Stress": {"idx_sigma_multiplier": 1.2},
    }
    transition = [[0.9, 0.1], [0.2, 0.8]]

    normalized, matrix, names = validate(regimes, transition)

    assert [regime["name"] for regime in normalized] == ["Calm", "Stress"]
    assert matrix == transition
    assert names == ["Calm", "Stress"]


def test_validate_regime_inputs_rejects_duplicate_names() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = [{"name": "Calm"}, {"name": "Calm"}]
    transition = [[0.6, 0.4], [0.3, 0.7]]

    with pytest.raises(ValueError, match="Regime names must be unique"):
        validate(regimes, transition)


def test_validate_regime_inputs_rejects_empty_regimes() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = []
    transition = []

    with pytest.raises(ValueError, match="Regimes must be a non-empty YAML/JSON list or mapping"):
        validate(regimes, transition)


def test_validate_regime_inputs_rejects_empty_transition() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = [{"name": "Calm"}]
    transition = []

    with pytest.raises(ValueError, match="Transition matrix must be a non-empty list of lists"):
        validate(regimes, transition)


def test_validate_regime_inputs_rejects_mapping_name_mismatch() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = {
        "Calm": {"name": "Stressed", "idx_sigma_multiplier": 0.8},
        "Stressed": {"idx_sigma_multiplier": 1.2},
    }
    transition = [[0.9, 0.1], [0.2, 0.8]]

    with pytest.raises(ValueError, match="Regime name 'Stressed' must match mapping key"):
        validate(regimes, transition)


def test_validate_regime_inputs_rejects_out_of_range_values() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = [{"name": "Calm"}, {"name": "Stress"}]
    transition = [[1.2, -0.2], [0.2, 0.8]]

    with pytest.raises(ValueError, match="Transition matrix row 1 values must be between 0 and 1"):
        validate(regimes, transition)


def test_validate_regime_inputs_rejects_non_square_transition() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = [{"name": "Calm"}, {"name": "Stress"}]
    transition = [[1.0], [0.2, 0.8]]

    with pytest.raises(ValueError, match="Transition matrix must be square"):
        validate(regimes, transition)


def test_validate_regime_inputs_strips_whitespace_names() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = {
        " Calm ": {"idx_sigma_multiplier": 0.8},
        "Stressed": {"idx_sigma_multiplier": 1.2},
    }
    transition = [[0.9, 0.1], [0.2, 0.8]]

    normalized, _matrix, names = validate(regimes, transition)

    assert names == ["Calm", "Stressed"]
    assert normalized[0]["name"] == "Calm"


def test_validate_regime_inputs_rejects_blank_name() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = [{"name": "   "}]
    transition = [[1.0]]

    with pytest.raises(ValueError, match="Regime #1 is missing a name"):
        validate(regimes, transition)


def test_validate_regime_inputs_rejects_blank_mapping_key() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = {"   ": {"idx_sigma_multiplier": 0.8}}
    transition = [[1.0]]

    with pytest.raises(ValueError, match="Regime name keys must be non-empty"):
        validate(regimes, transition)


def test_validate_yaml_dict_accepts_default_config() -> None:
    helpers = _load_helpers()
    build_yaml = helpers["_build_yaml_from_config"]
    validate = helpers["_validate_yaml_dict"]

    st.session_state.clear()
    try:
        config = get_default_config(AnalysisMode.RETURNS)
        yaml_dict = build_yaml(config)

        validate(yaml_dict)
    finally:
        st.session_state.clear()


def test_validate_yaml_dict_rejects_missing_terminal_shortfall() -> None:
    helpers = _load_helpers()
    build_yaml = helpers["_build_yaml_from_config"]
    validate = helpers["_validate_yaml_dict"]

    st.session_state.clear()
    try:
        config = get_default_config(AnalysisMode.RETURNS)
        yaml_dict = build_yaml(config)
        yaml_dict["risk_metrics"] = ["Return"]

        with pytest.raises(ValueError, match="terminal_ShortfallProb"):
            validate(yaml_dict)
    finally:
        st.session_state.clear()


def test_temp_yaml_file_writes_and_cleans_up() -> None:
    helpers = _load_helpers()
    temp_yaml = helpers["_temp_yaml_file"]

    payload = {"alpha": 1, "beta": 2}
    with temp_yaml(payload) as path:
        file_path = Path(path)
        assert file_path.exists()
        assert "alpha: 1" in file_path.read_text()

    assert not file_path.exists()
