from __future__ import annotations

from typing import Dict

import pytest

from pa_core.config import ConfigError, load_config


def base_config() -> Dict[str, object]:
    return {
        "N_SIMULATIONS": 1000,
        "N_MONTHS": 12,
        "analysis_mode": "returns",
        "risk_metrics": ["Return", "Risk", "ShortfallProb"],
    }


def test_financing_model_rejects_unknown_value() -> None:
    config_data = base_config()
    config_data["financing_model"] = "invalid"

    with pytest.raises(ValueError, match="financing_model must be one of"):
        load_config(config_data)


def test_financing_model_schedule_requires_path() -> None:
    config_data = base_config()
    config_data["financing_model"] = "schedule"

    with pytest.raises(ValueError, match="financing_schedule_path required"):
        load_config(config_data)


def test_backend_validation_rejects_unknown_backend() -> None:
    config_data = base_config()
    config_data["backend"] = "cuda"

    with pytest.raises(ValueError, match="backend must be one of"):
        load_config(config_data)


def test_analysis_mode_validation_rejects_unknown_mode() -> None:
    config_data = base_config()
    config_data["analysis_mode"] = "mystery"

    with pytest.raises(ValueError, match="analysis_mode must be one of"):
        load_config(config_data)


def test_share_weights_must_sum_to_one() -> None:
    config_data = base_config()
    config_data["w_beta_H"] = 0.7
    config_data["w_alpha_H"] = 0.7

    with pytest.raises(ValueError, match="w_beta_H and w_alpha_H must sum to 1"):
        load_config(config_data)


def test_theta_extpa_bounds_validation() -> None:
    config_data = base_config()
    config_data["theta_extpa"] = -0.1

    with pytest.raises(ValueError, match="theta_extpa must be between 0 and 1"):
        load_config(config_data)


def test_share_inputs_normalize_percent_values() -> None:
    config_data = base_config()
    config_data["active_share"] = 50.0
    config_data["theta_extpa"] = 80.0

    cfg = load_config(config_data)

    assert cfg.active_share == pytest.approx(0.5)
    assert cfg.theta_extpa == pytest.approx(0.8)


def test_share_inputs_accept_active_share_alias() -> None:
    config_data = base_config()
    config_data["Active share"] = 50.0

    cfg = load_config(config_data)

    assert cfg.active_share == pytest.approx(0.5)


def test_risk_metrics_must_include_shortfallprob() -> None:
    config_data = base_config()
    config_data["risk_metrics"] = ["Return", "Risk"]

    with pytest.raises(ValueError, match="risk_metrics must include ShortfallProb"):
        load_config(config_data)


def test_load_config_raises_for_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(missing_path)


def test_load_config_invalid_yaml_raises_config_error(tmp_path) -> None:
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("bad: [")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(bad_path)


def test_load_config_missing_required_field_raises_value_error() -> None:
    config_data = {
        "N_SIMULATIONS": 1000,
        "analysis_mode": "returns",
        "risk_metrics": ["Return", "Risk", "ShortfallProb"],
    }

    with pytest.raises(ValueError, match="Number of months"):
        load_config(config_data)
