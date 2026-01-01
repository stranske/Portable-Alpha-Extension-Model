import logging
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from pa_core.config import ModelConfig, annual_mean_to_monthly, load_config
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
    names = {agent.name for agent in cfg.agents}
    assert {"Base", "ExternalPA", "ActiveExt", "InternalPA"} <= names


def test_load_yaml_with_generic_agents(tmp_path):
    data = {
        "N_SIMULATIONS": 1000,
        "N_MONTHS": 6,
        "total_fund_capital": 300.0,
        "agents": [
            {
                "name": "Base",
                "capital": 300.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            },
            {
                "name": "CustomSleeve",
                "capital": 25.0,
                "beta_share": 0.1,
                "alpha_share": 0.05,
                "extra": {"active_return_volatility_target": 0.03},
            },
        ],
    }
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)
    assert len(cfg.agents) == 2
    assert {agent.name for agent in cfg.agents} == {"Base", "CustomSleeve"}


def test_load_yaml_with_mixed_agents(tmp_path):
    data = {
        "N_SIMULATIONS": 1000,
        "N_MONTHS": 6,
        "external_pa_capital": 100.0,
        "active_ext_capital": 50.0,
        "internal_pa_capital": 150.0,
        "total_fund_capital": 325.0,
        "agents": [
            {
                "name": "CustomSleeve",
                "capital": 25.0,
                "beta_share": 0.1,
                "alpha_share": 0.05,
                "extra": {},
            }
        ],
    }
    path = tmp_path / "mixed_agents.yaml"
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)
    names = {agent.name for agent in cfg.agents}
    assert {"Base", "ExternalPA", "ActiveExt", "InternalPA", "CustomSleeve"} <= names


def test_load_dict():
    data = {"N_SIMULATIONS": 1000, "N_MONTHS": 3, "mu_H": 0.05}  # Valid N_SIMULATIONS
    cfg = load_config(data)
    assert cfg.N_SIMULATIONS == 1000
    assert cfg.N_MONTHS == 3
    assert cfg.mu_H == annual_mean_to_monthly(0.05)


def test_model_config_minimal_inputs_use_defaults():
    data = {"N_SIMULATIONS": 1, "N_MONTHS": 1}
    cfg = ModelConfig(**data)
    assert cfg.total_fund_capital == 1000.0
    assert cfg.external_pa_capital == 0.0
    assert cfg.active_ext_capital == 0.0
    assert cfg.internal_pa_capital == 0.0
    assert cfg.w_beta_H == pytest.approx(0.5)
    assert cfg.w_alpha_H == pytest.approx(0.5)
    assert cfg.mu_H == annual_mean_to_monthly(0.04)
    assert cfg.agents and cfg.agents[0].name == "Base"


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


def test_agents_missing_benchmark():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "CustomSleeve",
                "capital": 10.0,
                "beta_share": 0.2,
                "alpha_share": 0.2,
                "extra": {},
            }
        ],
    }
    with pytest.raises(ValueError, match="benchmark agent named 'Base'"):
        ModelConfig(**data)


def test_agents_missing_benchmark_lists_existing_names():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "CustomSleeve",
                "capital": 10.0,
                "beta_share": 0.2,
                "alpha_share": 0.2,
                "extra": {},
            }
        ],
    }
    with pytest.raises(ValueError) as excinfo:
        ModelConfig(**data)
    assert "Existing agents: CustomSleeve." in str(excinfo.value)


def test_agents_missing_benchmark_without_convenience_fields():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "agents": [
            {
                "name": "CustomSleeve",
                "capital": 10.0,
                "beta_share": 0.2,
                "alpha_share": 0.2,
                "extra": {},
            }
        ],
    }
    with pytest.raises(ValueError, match="benchmark agent named 'Base'"):
        ModelConfig(**data)


def test_agents_empty_list_missing_benchmark():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [],
    }
    with pytest.raises(ValueError, match="benchmark agent named 'Base'"):
        ModelConfig(**data)


def test_agents_duplicate_names():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "Base",
                "capital": 80.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            },
            {
                "name": "Base",
                "capital": 10.0,
                "beta_share": 0.1,
                "alpha_share": 0.1,
                "extra": {},
            },
        ],
    }
    with pytest.raises(ValueError, match="agent names must be unique"):
        ModelConfig(**data)


def test_agents_duplicate_names_include_indices():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "Base",
                "capital": 80.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            },
            {
                "name": "Base",
                "capital": 10.0,
                "beta_share": 0.1,
                "alpha_share": 0.1,
                "extra": {},
            },
        ],
    }
    with pytest.raises(ValueError) as excinfo:
        ModelConfig(**data)
    assert "Base (indices [0, 1])" in str(excinfo.value)


def test_agents_multiple_benchmark_agents():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "Base",
                "capital": 80.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            },
            {
                "name": "Base",
                "capital": 10.0,
                "beta_share": 0.1,
                "alpha_share": 0.1,
                "extra": {},
            },
        ],
    }
    with pytest.raises(ValueError, match="benchmark agent named 'Base' is required; found 2"):
        ModelConfig(**data)


def test_agents_duplicate_base_in_mixed_config():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "external_pa_capital": 10.0,
        "w_beta_H": 0.6,
        "w_alpha_H": 0.4,
        "agents": [
            {
                "name": "Base",
                "capital": 100.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            }
        ],
    }
    cfg = ModelConfig(**data)
    agent_names = {agent.name for agent in cfg.agents}
    assert "Base" in agent_names
    assert "ExternalPA" in agent_names


def test_agents_negative_capital():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "Base",
                "capital": -1.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            }
        ],
    }
    with pytest.raises(ValueError, match="capital must be >= 0"):
        ModelConfig(**data)


def test_agents_share_bounds():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "Base",
                "capital": 100.0,
                "beta_share": 1.2,
                "alpha_share": 0.0,
                "extra": {},
            }
        ],
    }
    with pytest.raises(ValueError, match="beta_share must be between 0 and 1"):
        ModelConfig(**data)


def test_agents_alpha_share_bounds():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "Base",
                "capital": 100.0,
                "beta_share": 0.0,
                "alpha_share": 1.1,
                "extra": {},
            }
        ],
    }
    with pytest.raises(ValueError, match="alpha_share must be between 0 and 1"):
        ModelConfig(**data)


def test_agents_share_sum_limit():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "Base",
                "capital": 100.0,
                "beta_share": 0.7,
                "alpha_share": 0.5,
                "extra": {},
            }
        ],
    }
    with pytest.raises(ValueError, match="beta_share \\+ alpha_share must be <= 1"):
        ModelConfig(**data)


def test_agents_share_sum_limit_non_benchmark():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "Base",
                "capital": 100.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            },
            {
                "name": "CustomSleeve",
                "capital": 10.0,
                "beta_share": 0.8,
                "alpha_share": 0.3,
                "extra": {},
            },
        ],
    }
    with pytest.raises(ValueError, match="CustomSleeve.*beta_share \\+ alpha_share must be <= 1"):
        ModelConfig(**data)


def test_agents_total_capital_exceeds_fund():
    data = {
        "N_SIMULATIONS": 1,
        "N_MONTHS": 1,
        "total_fund_capital": 100.0,
        "agents": [
            {
                "name": "Base",
                "capital": 100.0,
                "beta_share": 0.6,
                "alpha_share": 0.4,
                "extra": {},
            },
            {
                "name": "CustomSleeve",
                "capital": 150.0,
                "beta_share": 0.1,
                "alpha_share": 0.1,
                "extra": {},
            },
        ],
    }
    with pytest.raises(
        ValueError, match="sum\\(non-benchmark agent capital\\) must be <= total_fund_capital"
    ):
        ModelConfig(**data)


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


def test_model_config_logs_transform_order(caplog: pytest.LogCaptureFixture) -> None:
    data = {
        "N_SIMULATIONS": 1000,
        "N_MONTHS": 12,
        "debug_transform_order": True,
    }
    with caplog.at_level(logging.INFO, logger="pa_core.config"):
        ModelConfig(**data)

    steps = [
        record.getMessage().split(": ", 1)[1]
        for record in caplog.records
        if record.name == "pa_core.config"
        and record.getMessage().startswith("ModelConfig transform: ")
    ]

    assert steps == [
        "parse_raw",
        "normalize_return_units",
        "normalize_share_inputs",
        "compile_agent_config",
        "check_financing_model",
        "check_capital",
        "check_return_distribution",
        "check_correlations",
        "check_shares",
        "check_analysis_mode",
        "check_vol_regime_window",
        "check_backend",
        "check_simulation_params",
    ]
