import numpy as np
import pandas as pd
import pytest

from pa_core import sweep as sweep_module
from pa_core.config import SweepConfig, SweepParameter, load_config
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


@pytest.mark.parametrize("explicit_seed", [None, 123])
def test_internal_pa_financing_draws_are_stable_when_combinations_reorder(
    monkeypatch, explicit_seed
):
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
            seed=explicit_seed,
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


@pytest.mark.parametrize("explicit_seed", [None, 123])
@pytest.mark.parametrize("financing_mode", ["broadcast", "per_path"])
@pytest.mark.parametrize("uncached_financing", [False, True])
def test_internal_pa_financing_isolated_from_standard_consumption(
    monkeypatch,
    explicit_seed,
    financing_mode,
    uncached_financing,
):
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={
            "N_SIMULATIONS": 8,
            "N_MONTHS": 6,
            "analysis_mode": "returns",
            "financing_mode": financing_mode,
            "internal_pa_financing_sigma_month": 0.01,
            "sweep": SweepConfig(
                parameters={
                    (
                        "internal_financing_mean_month" if uncached_financing else "theta_extpa"
                    ): SweepParameter(values=[0.0, 0.01])
                }
            ),
        }
    )
    combinations = (
        [
            {"internal_financing_mean_month": 0.0},
            {"internal_financing_mean_month": 0.01},
        ]
        if uncached_financing
        else [{"theta_extpa": 0.2}, {"theta_extpa": 0.8}]
    )
    idx = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01])
    real_draw = sweep_module.draw_financing_series
    real_resolve = sweep_module.resolve_internal_pa_financing_series

    def run(inject_standard_draws):
        captured = []
        draw_calls = 0

        def draw_with_optional_consumption(*args, **kwargs):
            nonlocal draw_calls
            draw_calls += 1
            matrices = real_draw(*args, **kwargs)
            if inject_standard_draws:
                kwargs["rngs"]["internal"].normal(size=137)
            return matrices

        def capture_internal_pa(*args, **kwargs):
            matrix = real_resolve(*args, **kwargs)
            captured.append(matrix.copy())
            return matrix

        monkeypatch.setattr(
            sweep_module,
            "generate_parameter_combinations",
            lambda _cfg: iter(combinations),
        )
        monkeypatch.setattr(sweep_module, "draw_financing_series", draw_with_optional_consumption)
        monkeypatch.setattr(
            sweep_module,
            "resolve_internal_pa_financing_series",
            capture_internal_pa,
        )
        sweep_module.run_parameter_sweep(
            cfg,
            idx,
            spawn_rngs(123, 1)[0],
            spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"]),
            seed=explicit_seed,
        )
        return captured, draw_calls

    baseline, baseline_calls = run(False)
    injected, injected_calls = run(True)

    assert baseline_calls == injected_calls == (2 if uncached_financing else 1)
    assert len(baseline) == len(injected) == 2
    for expected, actual in zip(baseline, injected, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_seeded_sweep_preserves_legacy_agent_rng_selection(monkeypatch):
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={
            "N_SIMULATIONS": 3,
            "N_MONTHS": 4,
            "analysis_mode": "returns",
            "internal_pa_financing_sigma_month": 0.01,
        }
    )
    real_spawn_agent_rngs = sweep_module.spawn_agent_rngs
    observed_legacy_order = []

    def spawn_with_observation(seed, agent_names, *, legacy_order=False):
        observed_legacy_order.append(legacy_order)
        return real_spawn_agent_rngs(seed, agent_names, legacy_order=legacy_order)

    monkeypatch.setattr(sweep_module, "spawn_agent_rngs", spawn_with_observation)
    monkeypatch.setattr(
        sweep_module,
        "generate_parameter_combinations",
        lambda _cfg: iter([{"theta_extpa": 0.2}]),
    )

    sweep_module.run_parameter_sweep(
        cfg,
        pd.Series([0.01, -0.02, 0.03, 0.0]),
        spawn_rngs(123, 1)[0],
        real_spawn_agent_rngs(
            123,
            ["internal", "external_pa", "active_ext"],
            legacy_order=True,
        ),
        seed=123,
        legacy_agent_rng=True,
    )

    assert observed_legacy_order == [True]
