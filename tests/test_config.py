from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from pa_core.config import ModelConfig, load_config
from pa_core.data.convert import convert

yaml: Any = pytest.importorskip("yaml")


def test_load_yaml(tmp_path):
    data = {
        "N_SIMULATIONS": 1000,  # Valid value
        "N_MONTHS": 6,
        "mu_H": 0.04,
        "sigma_H": 0.01,
        "external_pa_capital": 100.0,
        "active_ext_capital": 200.0,
        "internal_pa_capital": 300.0,
        "total_fund_capital": 1000.0,
    }
    path = tmp_path / "conf.yaml"
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)
    assert isinstance(cfg, ModelConfig)
    assert cfg.N_SIMULATIONS == 1000
    assert cfg.N_MONTHS == 6
    assert cfg.external_pa_capital == 100.0
    assert cfg.active_ext_capital == 200.0
    assert cfg.internal_pa_capital == 300.0
    assert cfg.total_fund_capital == 1000.0


def test_load_dict():
    data = {"N_SIMULATIONS": 1000, "N_MONTHS": 3, "mu_H": 0.05}  # Valid N_SIMULATIONS
    cfg = load_config(data)
    assert cfg.N_SIMULATIONS == 1000
    assert cfg.N_MONTHS == 3
    assert cfg.mu_H == 0.05


def test_model_config_normalizes_share_percentages():
    data = {"N_SIMULATIONS": 1, "N_MONTHS": 1, "w_beta_H": 60, "w_alpha_H": 40}
    cfg = ModelConfig(**data)
    assert cfg.w_beta_H == 0.6
    assert cfg.w_alpha_H == 0.4


def test_invalid_capital(tmp_path):
    data = {
        "N_SIMULATIONS": 1000,  # Valid value
        "N_MONTHS": 1,
        "external_pa_capital": 800.0,
        "active_ext_capital": 800.0,
        "internal_pa_capital": 800.0,
        "total_fund_capital": 1000.0,
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    try:
        load_config(path)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected validation failure")


def test_template_yaml_loads():
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "params_template.yml"
    cfg = load_config(cfg_path)
    assert isinstance(cfg, ModelConfig)


def test_csv_to_yaml_conversion(tmp_path):
    """Test CSV to YAML conversion using a generated CSV file."""
    # Create a minimal CSV file for testing conversion
    # The CSV loader expects:
    # - Columns named "Parameter" and "Value" (capitalized)
    # - Parameter names as human-readable aliases from ModelConfig
    csv_path = tmp_path / "test_params.csv"
    csv_content = dedent(
        """\
        Parameter,Value
        Number of simulations,1000
        Number of months,6
        In-House annual return (%),4.0
        In-House annual vol (%),1.0
        External PA capital (mm),100.0
        Active Extension capital (mm),200.0
        Internal PA capital (mm),300.0
        Total fund capital (mm),1000.0
    """
    )
    csv_path.write_text(csv_content)
    out_yaml = tmp_path / "out.yml"
    convert(csv_path, out_yaml)
    cfg = load_config(out_yaml)
    assert isinstance(cfg, ModelConfig)


def test_load_config_with_covariance_options(tmp_path):
    data = {
        "N_SIMULATIONS": 1000,
        "N_MONTHS": 6,
        "covariance_shrinkage": "ledoit_wolf",
        "vol_regime": "two_state",
        "vol_regime_window": 6,
    }
    path = tmp_path / "conf.yaml"
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)
    assert cfg.covariance_shrinkage == "ledoit_wolf"
    assert cfg.vol_regime == "two_state"
    assert cfg.vol_regime_window == 6


def test_model_config_rejects_out_of_bounds_correlations():
    data = {"N_SIMULATIONS": 1, "N_MONTHS": 1, "rho_idx_H": 1.5}
    with pytest.raises(ValueError, match="outside valid range"):
        ModelConfig(**data)
