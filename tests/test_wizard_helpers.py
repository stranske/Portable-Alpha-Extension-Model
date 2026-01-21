import runpy

import pytest

from pa_core.config import RegimeConfig
from pa_core.wizard_schema import RiskMetric


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


def test_validate_regime_inputs_rejects_non_square_transition() -> None:
    helpers = _load_helpers()
    validate = helpers["_validate_regime_inputs"]

    regimes = [{"name": "Calm"}, {"name": "Stress"}]
    transition = [[1.0], [0.2, 0.8]]

    with pytest.raises(ValueError, match="Transition matrix must be square"):
        validate(regimes, transition)
