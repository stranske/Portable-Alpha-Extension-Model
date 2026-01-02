from typing import Any

import numpy as np
import pytest

from pa_core.config import ModelConfig
from pa_core.sim.metrics import conditional_value_at_risk
from pa_core.sim.params import build_simulation_params
from pa_core.sim.paths import (
    _validate_correlation_matrix,
    draw_joint_returns,
    prepare_mc_universe,
    prepare_return_shocks,
    simulate_alpha_streams,
)


def _base_params() -> dict[str, Any]:
    cfg = ModelConfig(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        return_unit="monthly",
        mu_H=0.003,
        sigma_H=0.03,
        mu_E=0.0045,
        sigma_E=0.025,
        mu_M=0.0035,
        sigma_M=0.02,
        rho_idx_H=0.2,
        rho_idx_E=0.1,
        rho_idx_M=0.05,
        rho_H_E=0.12,
        rho_H_M=0.08,
        rho_E_M=0.07,
        return_distribution="normal",
        return_t_df=6.0,
        return_copula="gaussian",
    )
    return build_simulation_params(cfg, mu_idx=0.004, idx_sigma=0.02)


def _build_cov(params: dict[str, float | str]) -> np.ndarray:
    sigma = np.array(
        [
            params["idx_sigma_month"],
            params["default_sigma_H"],
            params["default_sigma_E"],
            params["default_sigma_M"],
        ],
        dtype=float,
    )
    corr = np.array(
        [
            [1.0, params["rho_idx_H"], params["rho_idx_E"], params["rho_idx_M"]],
            [params["rho_idx_H"], 1.0, params["rho_H_E"], params["rho_H_M"]],
            [params["rho_idx_E"], params["rho_H_E"], 1.0, params["rho_E_M"]],
            [params["rho_idx_M"], params["rho_H_M"], params["rho_E_M"], 1.0],
        ],
        dtype=float,
    )
    return corr * (sigma[:, None] * sigma[None, :])


def test_draw_joint_returns_variance_matches_sigma() -> None:
    n_sim, n_months = 2000, 24
    params = _base_params()
    rng = np.random.default_rng(123)
    r_beta, _, _, _ = draw_joint_returns(
        n_months=n_months,
        n_sim=n_sim,
        params=params,
        rng=rng,
    )
    assert r_beta.shape == (n_sim, n_months)
    sample_std = float(np.std(r_beta, ddof=1))
    assert np.isclose(sample_std, params["idx_sigma_month"], rtol=0.1)

    t_params = params.copy()
    t_params["return_distribution"] = "student_t"
    t_params["return_copula"] = "t"
    rng_t = np.random.default_rng(321)
    r_beta_t, _, _, _ = draw_joint_returns(
        n_months=n_months,
        n_sim=n_sim,
        params=t_params,
        rng=rng_t,
    )
    assert r_beta_t.shape == (n_sim, n_months)
    sample_std_t = float(np.std(r_beta_t, ddof=1))
    assert np.isclose(sample_std_t, t_params["idx_sigma_month"], rtol=0.1)


def test_simulate_alpha_streams_supports_student_t() -> None:
    params = _base_params()
    cov = _build_cov(params)
    n_obs = 3000
    rng_norm = np.random.default_rng(101)
    streams = simulate_alpha_streams(
        n_obs,
        cov,
        params["mu_idx_month"],
        params["default_mu_H"],
        params["default_mu_E"],
        params["default_mu_M"],
        rng=rng_norm,
    )
    assert streams.shape == (n_obs, 4)
    sample_std = float(np.std(streams[:, 0], ddof=1))
    assert np.isclose(sample_std, params["idx_sigma_month"], rtol=0.1)

    rng_t = np.random.default_rng(202)
    streams_t = simulate_alpha_streams(
        n_obs,
        cov,
        params["mu_idx_month"],
        params["default_mu_H"],
        params["default_mu_E"],
        params["default_mu_M"],
        return_distribution="student_t",
        return_t_df=6.0,
        return_copula="t",
        rng=rng_t,
    )
    assert streams_t.shape == (n_obs, 4)
    sample_std_t = float(np.std(streams_t[:, 0], ddof=1))
    assert np.isclose(sample_std_t, params["idx_sigma_month"], rtol=0.1)


def test_student_t_cvar_more_extreme_than_normal() -> None:
    n_sim, n_months = 5000, 12
    params = _base_params()
    normal_rng = np.random.default_rng(7)
    r_beta_norm, _, _, _ = draw_joint_returns(
        n_months=n_months,
        n_sim=n_sim,
        params=params,
        rng=normal_rng,
    )
    normal_cvar = conditional_value_at_risk(r_beta_norm, confidence=0.95)

    t_params = params.copy()
    t_params["return_distribution"] = "student_t"
    t_params["return_copula"] = "t"
    t_rng = np.random.default_rng(11)
    r_beta_t, _, _, _ = draw_joint_returns(
        n_months=n_months,
        n_sim=n_sim,
        params=t_params,
        rng=t_rng,
    )
    t_cvar = conditional_value_at_risk(r_beta_t, confidence=0.95)
    assert t_cvar < normal_cvar


def test_draw_joint_returns_allows_per_series_distributions() -> None:
    n_sim, n_months = 6000, 12
    params = _base_params()
    params.update(
        {
            "default_mu_H": params["mu_idx_month"],
            "default_sigma_H": params["idx_sigma_month"],
            "rho_idx_H": 0.0,
            "rho_idx_E": 0.0,
            "rho_idx_M": 0.0,
            "rho_H_E": 0.0,
            "rho_H_M": 0.0,
            "rho_E_M": 0.0,
            "return_distribution_idx": "normal",
            "return_distribution_H": "student_t",
            "return_distribution_E": "normal",
            "return_distribution_M": "normal",
        }
    )
    rng = np.random.default_rng(9)
    r_beta, r_H, _, _ = draw_joint_returns(
        n_months=n_months,
        n_sim=n_sim,
        params=params,
        rng=rng,
    )
    normal_cvar = conditional_value_at_risk(r_beta, confidence=0.95)
    t_cvar = conditional_value_at_risk(r_H, confidence=0.95)
    assert t_cvar < normal_cvar


def test_prepare_mc_universe_supports_per_series_student_t() -> None:
    n_sim, n_months = 4000, 12
    sigma = 0.02
    cov = np.diag([sigma**2, sigma**2, sigma**2, sigma**2])
    rng = np.random.default_rng(2024)
    with pytest.warns(DeprecationWarning, match="prepare_mc_universe is deprecated"):
        sims = prepare_mc_universe(
            N_SIMULATIONS=n_sim,
            N_MONTHS=n_months,
            mu_idx=0.0,
            mu_H=0.0,
            mu_E=0.0,
            mu_M=0.0,
            cov_mat=cov,
            return_distribution="normal",
            return_t_df=6.0,
            return_copula="gaussian",
            return_distributions=("normal", "student_t", "normal", "normal"),
            rng=rng,
        )
    normal_cvar = conditional_value_at_risk(sims[:, :, 0], confidence=0.95)
    t_cvar = conditional_value_at_risk(sims[:, :, 1], confidence=0.95)
    assert t_cvar < normal_cvar


def test_draw_joint_returns_matches_prepared_shocks() -> None:
    n_sim, n_months = 500, 6
    params = _base_params()
    rng_shocks = np.random.default_rng(123)
    shocks = prepare_return_shocks(
        n_months=n_months,
        n_sim=n_sim,
        params=params,
        rng=rng_shocks,
    )
    shocked = draw_joint_returns(
        n_months=n_months,
        n_sim=n_sim,
        params=params,
        shocks=shocks,
    )
    repeated = draw_joint_returns(
        n_months=n_months,
        n_sim=n_sim,
        params=params,
        shocks=shocks,
    )
    for left, right in zip(shocked, repeated):
        np.testing.assert_allclose(left, right)


def test_draw_joint_returns_passes_through_valid_correlation() -> None:
    params = _base_params()
    rng = np.random.default_rng(1234)
    draw_joint_returns(
        n_months=3,
        n_sim=10,
        params=params,
        rng=rng,
    )
    info = params.get("_correlation_repair_info")
    assert isinstance(info, dict)
    assert info["repair_applied"] is False
    assert info["method"] == "none"


def test_draw_joint_returns_rejects_out_of_range_correlations() -> None:
    params = _base_params()
    params["rho_idx_H"] = 1.25
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="within \\[-1, 1\\]"):
        draw_joint_returns(
            n_months=2,
            n_sim=5,
            params=params,
            rng=rng,
        )


def test_validate_correlation_matrix_rejects_bad_diagonal() -> None:
    corr = np.eye(3)
    corr[1, 1] = 0.9
    with pytest.raises(ValueError, match="diagonal must be 1"):
        _validate_correlation_matrix(corr)


def test_draw_joint_returns_repairs_non_psd_correlation() -> None:
    params = _base_params()
    params.update(
        {
            "rho_idx_H": 0.9,
            "rho_idx_E": 0.9,
            "rho_idx_M": 0.0,
            "rho_H_E": -0.9,
            "rho_H_M": 0.0,
            "rho_E_M": 0.0,
            "correlation_repair_mode": "warn_fix",
        }
    )
    rng = np.random.default_rng(7)
    draw_joint_returns(
        n_months=4,
        n_sim=8,
        params=params,
        rng=rng,
    )
    info = params.get("_correlation_repair_info")
    assert isinstance(info, dict)
    assert info["repair_applied"] is True
    assert "eigen_clip" in info["method"]
    assert info["min_eigenvalue_before"] < 0.0
    assert info["min_eigenvalue_after"] >= -1e-10
