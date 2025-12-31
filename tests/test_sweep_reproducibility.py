import pandas as pd

from pa_core import sweep as sweep_module
from pa_core.config import load_config
from pa_core.random import spawn_agent_rngs, spawn_rngs


def _run_sweep(cfg, idx, seed):
    rng_returns = spawn_rngs(seed, 1)[0]
    fin_rngs = spawn_agent_rngs(seed, ["internal", "external_pa", "active_ext"])
    return sweep_module.run_parameter_sweep(cfg, idx, rng_returns, fin_rngs)


def test_sweep_reproducible_with_master_seed(monkeypatch):
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3, "analysis_mode": "returns"}
    )
    idx = pd.Series([0.01, -0.02, 0.015])
    combos = [
        {
            "mu_H": cfg.mu_H,
            "sigma_H": cfg.sigma_H,
            "mu_E": cfg.mu_E,
            "sigma_E": cfg.sigma_E,
        },
        {
            "mu_H": cfg.mu_H * 1.1,
            "sigma_H": cfg.sigma_H,
            "mu_E": cfg.mu_E,
            "sigma_E": cfg.sigma_E,
        },
    ]

    monkeypatch.setattr(
        sweep_module,
        "generate_parameter_combinations",
        lambda _cfg: iter(combos),
    )

    res1 = _run_sweep(cfg, idx, seed=123)
    res2 = _run_sweep(cfg, idx, seed=123)

    assert len(res1) == len(res2)
    for left, right in zip(res1, res2):
        assert left["parameters"] == right["parameters"]
        pd.testing.assert_frame_equal(left["summary"], right["summary"])


def test_sweep_seed_resets_rng_state(monkeypatch):
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 2, "N_MONTHS": 3, "analysis_mode": "returns"}
    )
    idx = pd.Series([0.01, -0.02, 0.015])
    combos = [
        {
            "mu_H": cfg.mu_H,
            "sigma_H": cfg.sigma_H,
            "mu_E": cfg.mu_E,
            "sigma_E": cfg.sigma_E,
        },
        {
            "mu_H": cfg.mu_H * 1.1,
            "sigma_H": cfg.sigma_H,
            "mu_E": cfg.mu_E,
            "sigma_E": cfg.sigma_E,
        },
    ]

    monkeypatch.setattr(
        sweep_module,
        "generate_parameter_combinations",
        lambda _cfg: iter(combos),
    )

    rng_returns = spawn_rngs(999, 1)[0]
    fin_rngs = spawn_agent_rngs(999, ["internal", "external_pa", "active_ext"])
    rng_returns.normal(size=5)
    for rng in fin_rngs.values():
        rng.normal(size=5)

    res1 = sweep_module.run_parameter_sweep(
        cfg, idx, rng_returns, fin_rngs, seed=123
    )

    rng_returns_2 = spawn_rngs(555, 1)[0]
    fin_rngs_2 = spawn_agent_rngs(555, ["internal", "external_pa", "active_ext"])
    res2 = sweep_module.run_parameter_sweep(
        cfg, idx, rng_returns_2, fin_rngs_2, seed=123
    )

    assert len(res1) == len(res2)
    for left, right in zip(res1, res2):
        assert left["parameters"] == right["parameters"]
        pd.testing.assert_frame_equal(left["summary"], right["summary"])
