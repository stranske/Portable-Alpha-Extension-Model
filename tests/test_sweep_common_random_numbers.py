import pandas as pd

from pa_core import sweep as sweep_module
from pa_core.config import load_config
from pa_core.random import spawn_agent_rngs, spawn_rngs


def test_duplicate_combinations_share_random_draws(monkeypatch):
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 3, "N_MONTHS": 4, "analysis_mode": "returns"}
    )
    idx = pd.Series([0.01, -0.02, 0.03, 0.0])
    overrides = {
        "mu_H": cfg.mu_H,
        "sigma_H": cfg.sigma_H,
        "mu_E": cfg.mu_E,
        "sigma_E": cfg.sigma_E,
    }
    combos = [overrides.copy(), overrides.copy()]

    monkeypatch.setattr(
        sweep_module,
        "generate_parameter_combinations",
        lambda _cfg: iter(combos),
    )

    rng_returns = spawn_rngs(123, 1)[0]
    fin_rngs = spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"])
    results = sweep_module.run_parameter_sweep(cfg, idx, rng_returns, fin_rngs)

    assert len(results) == 2
    pd.testing.assert_frame_equal(results[0]["summary"], results[1]["summary"])


def test_duplicate_combinations_share_random_draws_with_seed(monkeypatch):
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={"N_SIMULATIONS": 3, "N_MONTHS": 4, "analysis_mode": "returns"}
    )
    idx = pd.Series([0.01, -0.02, 0.03, 0.0])
    overrides = {
        "mu_H": cfg.mu_H,
        "sigma_H": cfg.sigma_H,
        "mu_E": cfg.mu_E,
        "sigma_E": cfg.sigma_E,
    }
    combos = [overrides.copy(), overrides.copy()]

    monkeypatch.setattr(
        sweep_module,
        "generate_parameter_combinations",
        lambda _cfg: iter(combos),
    )

    rng_returns = spawn_rngs(999, 1)[0]
    fin_rngs = spawn_agent_rngs(999, ["internal", "external_pa", "active_ext"])
    rng_returns.normal(size=3)
    for rng in fin_rngs.values():
        rng.normal(size=3)

    results = sweep_module.run_parameter_sweep(cfg, idx, rng_returns, fin_rngs, seed=123)

    assert len(results) == 2
    pd.testing.assert_frame_equal(results[0]["summary"], results[1]["summary"])


def test_internal_pa_financing_draws_are_stable_when_combinations_reorder(monkeypatch):
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={
            "N_SIMULATIONS": 8,
            "N_MONTHS": 6,
            "analysis_mode": "returns",
            "internal_pa_financing_sigma_month": 0.01,
        }
    )
    idx = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01])
    combinations = [{"theta_extpa": 0.2}, {"theta_extpa": 0.8}]

    def run(order):
        monkeypatch.setattr(
            sweep_module,
            "generate_parameter_combinations",
            lambda _cfg: iter(order),
        )
        results = sweep_module.run_parameter_sweep(
            cfg,
            idx,
            spawn_rngs(123, 1)[0],
            spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"]),
        )
        return {
            result["parameters"]["theta_extpa"]: result["summary"]
            .loc[lambda frame: frame["Agent"] == "InternalPA"]
            .reset_index(drop=True)
            for result in results
        }

    forward = run(combinations)
    reverse = run(list(reversed(combinations)))

    for theta in (0.2, 0.8):
        pd.testing.assert_frame_equal(forward[theta], reverse[theta])


def test_internal_pa_financing_inputs_still_change_results(monkeypatch):
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={
            "N_SIMULATIONS": 8,
            "N_MONTHS": 6,
            "analysis_mode": "returns",
            "internal_pa_financing_sigma_month": 0.01,
        }
    )
    combinations = [
        {"internal_pa_financing_mean_month": 0.0},
        {"internal_pa_financing_mean_month": 0.01},
    ]
    monkeypatch.setattr(
        sweep_module,
        "generate_parameter_combinations",
        lambda _cfg: iter(combinations),
    )

    results = sweep_module.run_parameter_sweep(
        cfg,
        pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01]),
        spawn_rngs(123, 1)[0],
        spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"]),
    )
    internal_pa_returns = [
        result["summary"]
        .loc[lambda frame: frame["Agent"] == "InternalPA", "terminal_AnnReturn"]
        .iloc[0]
        for result in results
    ]

    assert internal_pa_returns[1] < internal_pa_returns[0]
