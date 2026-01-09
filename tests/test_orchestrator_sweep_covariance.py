import numpy as np
import pandas as pd

from pa_core.config import load_config
from pa_core.orchestrator import SimulatorOrchestrator
from pa_core.random import spawn_agent_rngs, spawn_rngs
from pa_core.sweep import run_parameter_sweep


def test_orchestrator_and_sweep_use_covariance_implied_params(monkeypatch) -> None:
    sigma = np.array([0.2, 0.3, 0.4, 0.5], dtype=float)
    corr = np.array(
        [
            [1.0, 0.1, 0.2, 0.3],
            [0.1, 1.0, 0.4, 0.5],
            [0.2, 0.4, 1.0, 0.6],
            [0.3, 0.5, 0.6, 1.0],
        ],
        dtype=float,
    )
    cov = np.outer(sigma, sigma) * corr

    def fake_build_cov_matrix(*_args, **_kwargs):
        return cov

    monkeypatch.setattr("pa_core.sim.covariance.build_cov_matrix", fake_build_cov_matrix)
    monkeypatch.setattr("pa_core.sweep.build_cov_matrix", fake_build_cov_matrix)

    def fake_draw_financing_series(*, n_months, n_sim, **_kwargs):
        zeros = np.zeros((n_months, n_sim))
        return zeros, zeros, zeros

    def fake_simulate_agents(*_args, **_kwargs):
        return {"Base": np.zeros((1, 1))}

    def fake_summary_table(*_args, **_kwargs):
        return pd.DataFrame(
            {
                "Agent": ["Base"],
                "terminal_AnnReturn": [0.0],
                "monthly_AnnVol": [0.0],
                "monthly_VaR": [0.0],
                "monthly_CVaR": [0.0],
                "terminal_CVaR": [0.0],
                "monthly_MaxDD": [0.0],
                "monthly_TimeUnderWater": [0.0],
                "monthly_BreachProb": [0.0],
                "monthly_BreachCountPath0": [0.0],
                "terminal_ShortfallProb": [0.0],
                "monthly_TE": [0.0],
            }
        )

    monkeypatch.setattr("pa_core.sim.draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr("pa_core.sweep.draw_financing_series", fake_draw_financing_series)
    monkeypatch.setattr("pa_core.simulations.simulate_agents", fake_simulate_agents)
    monkeypatch.setattr("pa_core.sweep.simulate_agents", fake_simulate_agents)
    monkeypatch.setattr("pa_core.sim.metrics.summary_table", fake_summary_table)
    monkeypatch.setattr("pa_core.sweep.summary_table", fake_summary_table)

    orch_params: dict[str, object] = {}
    sweep_params: dict[str, object] = {}

    def capture_orch_draw_joint_returns(
        *, n_months, n_sim, params, rng=None, shocks=None, regime_paths=None, regime_params=None
    ):
        orch_params["params"] = dict(params)
        zeros = np.zeros((n_months, n_sim))
        return zeros, zeros, zeros, zeros

    def capture_sweep_draw_joint_returns(
        *, n_months, n_sim, params, rng=None, shocks=None, regime_paths=None, regime_params=None
    ):
        if "params" not in sweep_params:
            sweep_params["params"] = dict(params)
        zeros = np.zeros((n_months, n_sim))
        return zeros, zeros, zeros, zeros

    monkeypatch.setattr("pa_core.sim.draw_joint_returns", capture_orch_draw_joint_returns)
    monkeypatch.setattr("pa_core.sweep.draw_joint_returns", capture_sweep_draw_joint_returns)

    cfg = load_config(
        {
            "N_SIMULATIONS": 5,
            "N_MONTHS": 2,
            "financing_mode": "broadcast",
            "analysis_mode": "returns",
            "risk_metrics": ["terminal_ShortfallProb"],
            "covariance_shrinkage": "ledoit_wolf",
            "in_house_return_min_pct": 2.0,
            "in_house_return_max_pct": 2.0,
            "in_house_return_step_pct": 1.0,
            "in_house_vol_min_pct": 1.0,
            "in_house_vol_max_pct": 1.0,
            "in_house_vol_step_pct": 1.0,
            "alpha_ext_return_min_pct": 1.0,
            "alpha_ext_return_max_pct": 1.0,
            "alpha_ext_return_step_pct": 1.0,
            "alpha_ext_vol_min_pct": 2.0,
            "alpha_ext_vol_max_pct": 2.0,
            "alpha_ext_vol_step_pct": 1.0,
        }
    )
    idx_series = pd.Series([0.01, 0.02, 0.015, 0.03])

    orch = SimulatorOrchestrator(cfg, idx_series)
    orch.run(seed=123)

    rng_returns = spawn_rngs(123, 1)[0]
    fin_rngs = spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"])
    run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)

    assert "params" in orch_params
    assert "params" in sweep_params

    sigma_vec = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(sigma_vec, sigma_vec)
    corr_mat = np.divide(cov, denom, out=np.eye(cov.shape[0]), where=denom != 0.0)

    expected_mu = float(idx_series.mean())
    expected = {
        "mu_idx_month": expected_mu,
        "idx_sigma_month": sigma_vec[0],
        "default_sigma_H": sigma_vec[1],
        "default_sigma_E": sigma_vec[2],
        "default_sigma_M": sigma_vec[3],
        "rho_idx_H": float(corr_mat[0, 1]),
        "rho_idx_E": float(corr_mat[0, 2]),
        "rho_idx_M": float(corr_mat[0, 3]),
        "rho_H_E": float(corr_mat[1, 2]),
        "rho_H_M": float(corr_mat[1, 3]),
        "rho_E_M": float(corr_mat[2, 3]),
    }

    for key, value in expected.items():
        assert orch_params["params"][key] == value
        assert sweep_params["params"][key] == value
