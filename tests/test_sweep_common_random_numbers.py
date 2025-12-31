import pandas as pd

from pa_core.config import load_config
from pa_core.random import spawn_agent_rngs, spawn_rngs
from pa_core import sweep as sweep_module


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
