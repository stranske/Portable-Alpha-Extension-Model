import numpy as np

from pa_core.sim.metrics import conditional_value_at_risk
from pa_core.sim.paths import draw_joint_returns, simulate_alpha_streams


def _base_params() -> dict[str, float | str]:
    return {
        "mu_idx_month": 0.004,
        "default_mu_H": 0.003,
        "default_mu_E": 0.0045,
        "default_mu_M": 0.0035,
        "idx_sigma_month": 0.02,
        "default_sigma_H": 0.03,
        "default_sigma_E": 0.025,
        "default_sigma_M": 0.02,
        "rho_idx_H": 0.2,
        "rho_idx_E": 0.1,
        "rho_idx_M": 0.05,
        "rho_H_E": 0.12,
        "rho_H_M": 0.08,
        "rho_E_M": 0.07,
        "return_distribution": "normal",
        "return_t_df": 6.0,
        "return_copula": "gaussian",
    }


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
