import pandas as pd

from pa_core.config import load_config
from pa_core.sweep import SweepRunner


def test_sweep_runner_calls_run_parameter_sweep(monkeypatch) -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml")
    idx = pd.Series([0.01, 0.02, -0.01])
    captured: dict[str, object] = {}

    def fake_run_parameter_sweep(cfg_arg, idx_arg, rng_returns, fin_rngs, seed=None, progress=None):
        captured["cfg"] = cfg_arg
        captured["idx"] = idx_arg
        captured["seed"] = seed
        captured["progress"] = progress
        return [{"combination_id": 0, "parameters": {}, "summary": pd.DataFrame()}]

    monkeypatch.setattr("pa_core.sweep.run_parameter_sweep", fake_run_parameter_sweep)

    runner = SweepRunner(cfg, idx, seed=123)
    results = runner.run()

    assert results[0]["combination_id"] == 0
    assert captured["cfg"] is cfg
    assert isinstance(captured["idx"], pd.Series)
    assert captured["seed"] is not None
