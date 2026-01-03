import pandas as pd

from pa_core.config import load_config
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
