from __future__ import annotations

from typing import Dict

import pytest

from pa_core.config import load_config


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


def test_risk_metrics_must_include_shortfallprob() -> None:
    config_data = base_config()
    config_data["risk_metrics"] = ["Return", "Risk"]

    with pytest.raises(ValueError, match="risk_metrics must include ShortfallProb"):
        load_config(config_data)
