import pandas as pd

from pa_core.config import RegimeConfig, load_config
from pa_core.facade import RunOptions, run_sweep


def test_run_sweep_returns_artifacts() -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])
    sweep_params = {
        "analysis_mode": "vol_mult",
        "sd_multiple_min": 1.0,
        "sd_multiple_max": 1.0,
        "sd_multiple_step": 1.0,
    }

    artifacts = run_sweep(cfg, idx, sweep_params, RunOptions(seed=123))

    assert artifacts.summary is not None
    assert not artifacts.summary.empty
    assert len(artifacts.results) == 1
    assert "combination_id" in artifacts.summary.columns
    assert "sigma_H" in artifacts.summary.columns
    assert artifacts.manifest is not None
    assert artifacts.manifest["seed"] == 123
    assert set(artifacts.manifest["substream_ids"]) == {
        "internal",
        "external_pa",
        "active_ext",
    }


def test_run_sweep_applies_config_overrides() -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])
    sweep_params = {
        "analysis_mode": "vol_mult",
        "sd_multiple_min": 1.0,
        "sd_multiple_max": 1.0,
        "sd_multiple_step": 1.0,
    }

    artifacts = run_sweep(
        cfg,
        idx,
        sweep_params,
        RunOptions(seed=123, config_overrides={"N_SIMULATIONS": 1}),
    )

    assert artifacts.config.N_SIMULATIONS == 1


def test_run_sweep_applies_regime_switching() -> None:
    base_cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={
            "N_SIMULATIONS": 128,
            "N_MONTHS": 12,
            "return_unit": "monthly",
            "financing_mode": "broadcast",
            "sigma_H": 0.02,
            "sigma_E": 0.02,
            "sigma_M": 0.02,
            "rho_idx_H": 0.1,
            "rho_idx_E": 0.1,
            "rho_idx_M": 0.1,
            "rho_H_E": 0.2,
            "rho_H_M": 0.2,
            "rho_E_M": 0.2,
        }
    )
    regime_cfg = base_cfg.model_copy(
        update={
            "regimes": [
                RegimeConfig(name="calm"),
                RegimeConfig(
                    name="stress",
                    idx_sigma_multiplier=2.0,
                    sigma_H=0.05,
                    sigma_E=0.05,
                    sigma_M=0.05,
                    rho_idx_H=0.8,
                    rho_idx_E=0.8,
                    rho_idx_M=0.8,
                    rho_H_E=0.85,
                    rho_H_M=0.85,
                    rho_E_M=0.85,
                ),
            ],
            "regime_transition": [[0.8, 0.2], [0.2, 0.8]],
            "regime_start": "calm",
        }
    )
    idx = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005, -0.01] * 2)
    sweep_params = {
        "analysis_mode": "vol_mult",
        "sd_multiple_min": 1.0,
        "sd_multiple_max": 1.0,
        "sd_multiple_step": 1.0,
    }

    without_regimes = run_sweep(base_cfg, idx, sweep_params, RunOptions(seed=123))
    with_regimes = run_sweep(regime_cfg, idx, sweep_params, RunOptions(seed=123))

    assert len(with_regimes.results) == len(without_regimes.results) == 1
    assert not with_regimes.summary.equals(without_regimes.summary)
