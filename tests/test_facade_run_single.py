import pandas as pd

from pa_core.config import load_config
from pa_core.facade import RunOptions, run_single


def test_run_single_returns_artifacts() -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])

    artifacts = run_single(cfg, idx, RunOptions(seed=123))

    assert artifacts.summary is not None
    assert not artifacts.summary.empty
    assert artifacts.config.N_SIMULATIONS == 2
    assert set(artifacts.returns) == set(artifacts.raw_returns)
    assert artifacts.manifest is not None
    assert artifacts.manifest["seed"] == 123
    assert set(artifacts.manifest["substream_ids"]) == {
        "internal",
        "external_pa",
        "active_ext",
    }


def test_run_single_applies_config_overrides() -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3}
    )
    idx = pd.Series([0.01, -0.02, 0.015])

    artifacts = run_single(cfg, idx, RunOptions(seed=123, config_overrides={"N_SIMULATIONS": 1}))

    assert artifacts.config.N_SIMULATIONS == 1
