import numpy as np
import pandas as pd

from pa_core import sweep as sweep_module
from pa_core.config import load_config
from pa_core.random import spawn_agent_rngs, spawn_rngs


def test_sweep_reuses_base_covariance_for_static_returns(monkeypatch) -> None:
    cfg = load_config("examples/scenarios/my_first_scenario.yml").model_copy(
        update={
            "analysis_mode": "capital",
            "N_SIMULATIONS": 2,
            "N_MONTHS": 3,
            "max_external_combined_pct": 10.0,
            "external_step_size_pct": 10.0,
        }
    )
    idx = pd.Series([0.01, -0.02, 0.03])
    combos = [
        {
            "external_pa_capital": 100.0,
            "active_ext_capital": 0.0,
            "internal_pa_capital": 200.0,
        },
        {
            "external_pa_capital": 120.0,
            "active_ext_capital": 0.0,
            "internal_pa_capital": 180.0,
        },
    ]

    monkeypatch.setattr(
        sweep_module, "generate_parameter_combinations", lambda _cfg: iter(combos)
    )

    cov_calls = {"count": 0}
    real_build_cov = sweep_module.build_cov_matrix

    def counting_build_cov(*args, **kwargs):
        cov_calls["count"] += 1
        return real_build_cov(*args, **kwargs)

    def fake_draw_joint_returns(*, n_months, n_sim, **_kwargs):
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros, zeros

    def fake_draw_financing_series(*, n_months, n_sim, **_kwargs):
        zeros = np.zeros((n_sim, n_months))
        return zeros, zeros, zeros

    def fake_simulate_agents(*_args, **_kwargs):
        return {"Base": np.zeros((cfg.N_SIMULATIONS, cfg.N_MONTHS))}

    def fake_summary_table(_returns, benchmark="Base"):
        return pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.0]})

    monkeypatch.setattr(sweep_module, "build_cov_matrix", counting_build_cov)
    monkeypatch.setattr(sweep_module, "draw_joint_returns", fake_draw_joint_returns)
    monkeypatch.setattr(sweep_module, "draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr(sweep_module, "simulate_agents", fake_simulate_agents)
    monkeypatch.setattr(sweep_module, "summary_table", fake_summary_table)
    monkeypatch.setattr(sweep_module, "build_from_config", lambda _cfg: [])

    rng_returns = spawn_rngs(123, 1)[0]
    fin_rngs = spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"])
    sweep_module.run_parameter_sweep(
        cfg, idx, rng_returns, fin_rngs, progress=lambda *_args: None
    )

    assert cov_calls["count"] == 1
