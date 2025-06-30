from __future__ import annotations

from typing import Optional, Any
import numpy as npt

from ..backend import xp as np
from numpy.typing import NDArray

__all__ = [
    "simulate_financing",
    "prepare_mc_universe",
    "draw_joint_returns",
    "draw_financing_series",
    "simulate_alpha_streams",
]


def simulate_financing(
    T: int,
    financing_mean: float,
    financing_sigma: float,
    spike_prob: float,
    spike_factor: float,
    *,
    seed: Optional[int] = None,
    n_scenarios: int = 1,
    rng: Optional[npt.random.Generator] = None,
) -> NDArray[Any]:
    """Vectorised financing spread simulation with optional spikes."""
    if T <= 0:
        raise ValueError("T must be positive")
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    rng = np.random.default_rng(seed) if rng is None else rng
    base = rng.normal(loc=financing_mean, scale=financing_sigma, size=(n_scenarios, T))
    jumps = (rng.random(size=(n_scenarios, T)) < spike_prob) * (spike_factor * financing_sigma)
    out = np.clip(base + jumps, 0.0, None)
    return out[0] if n_scenarios == 1 else out


def prepare_mc_universe(
    *,
    N_SIMULATIONS: int,
    N_MONTHS: int,
    mu_idx: float,
    mu_H: float,
    mu_E: float,
    mu_M: float,
    cov_mat: NDArray[Any],
    seed: Optional[int] = None,
    rng: Optional[npt.random.Generator] = None,
) -> NDArray[Any]:
    """Return stacked draws of (index, H, E, M) returns."""
    if N_SIMULATIONS <= 0 or N_MONTHS <= 0:
        raise ValueError("N_SIMULATIONS and N_MONTHS must be positive")
    if cov_mat.shape != (4, 4):
        raise ValueError("cov_mat must be 4×4 and ordered as [idx, H, E, M]")
    rng = np.random.default_rng(seed) if rng is None else rng
    z = rng.standard_normal(size=(N_SIMULATIONS, N_MONTHS, 4))
    try:
        L = np.linalg.cholesky(cov_mat / 12.0)
    except np.linalg.LinAlgError:
        eps = 1e-12
        L = np.linalg.cholesky(cov_mat / 12.0 + np.eye(4) * eps)
    mu = np.array([mu_idx, mu_H, mu_E, mu_M]) / 12.0
    return z @ L.T + mu


def draw_joint_returns(
    *,
    n_months: int,
    n_sim: int,
    params: dict,
    rng: Optional[npt.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised draw of monthly returns for (beta, H, E, M)."""
    if rng is None:
        rng = np.random.default_rng()
    μ_idx = params["mu_idx_month"]
    μ_H = params["default_mu_H"]
    μ_E = params["default_mu_E"]
    μ_M = params["default_mu_M"]
    σ_idx = params["idx_sigma_month"]
    σ_H = params["default_sigma_H"]
    σ_E = params["default_sigma_E"]
    σ_M = params["default_sigma_M"]
    ρ_idx_H = params["rho_idx_H"]
    ρ_idx_E = params["rho_idx_E"]
    ρ_idx_M = params["rho_idx_M"]
    ρ_H_E = params["rho_H_E"]
    ρ_H_M = params["rho_H_M"]
    ρ_E_M = params["rho_E_M"]
    Σ = np.array(
        [
            [σ_idx**2, ρ_idx_H * σ_idx * σ_H, ρ_idx_E * σ_idx * σ_E, ρ_idx_M * σ_idx * σ_M],
            [ρ_idx_H * σ_idx * σ_H, σ_H**2, ρ_H_E * σ_H * σ_E, ρ_H_M * σ_H * σ_M],
            [ρ_idx_E * σ_idx * σ_E, ρ_H_E * σ_H * σ_E, σ_E**2, ρ_E_M * σ_E * σ_M],
            [ρ_idx_M * σ_idx * σ_M, ρ_H_M * σ_H * σ_M, ρ_E_M * σ_E * σ_M, σ_M**2],
        ]
    )
    μ = np.array([μ_idx, μ_H, μ_E, μ_M])
    sims = rng.multivariate_normal(mean=μ, cov=Σ, size=(n_sim, n_months))
    r_beta = sims[:, :, 0]
    r_H = sims[:, :, 1]
    r_E = sims[:, :, 2]
    r_M = sims[:, :, 3]
    return r_beta, r_H, r_E, r_M


def draw_financing_series(
    *,
    n_months: int,
    n_sim: int,
    params: dict,
    rng: Optional[npt.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three matrices of monthly financing spreads."""
    if rng is None:
        rng = np.random.default_rng()

    def _sim(mean_key: str, sigma_key: str, p_key: str, k_key: str) -> NDArray[Any]:
        mean = params[mean_key]
        sigma = params[sigma_key]
        p = params[p_key]
        k = params[k_key]
        vec = simulate_financing(n_months, mean, sigma, p, k, n_scenarios=1, rng=rng)[0]
        return np.broadcast_to(vec, (n_sim, n_months))

    f_int_mat = _sim(
        "internal_financing_mean_month",
        "internal_financing_sigma_month",
        "internal_spike_prob",
        "internal_spike_factor",
    )
    f_ext_pa_mat = _sim(
        "ext_pa_financing_mean_month",
        "ext_pa_financing_sigma_month",
        "ext_pa_spike_prob",
        "ext_pa_spike_factor",
    )
    f_act_ext_mat = _sim(
        "act_ext_financing_mean_month",
        "act_ext_financing_sigma_month",
        "act_ext_spike_prob",
        "act_ext_spike_factor",
    )
    return f_int_mat, f_ext_pa_mat, f_act_ext_mat


def simulate_alpha_streams(T: int, cov: NDArray[Any], mu_idx: float, mu_H: float, mu_E: float, mu_M: float) -> NDArray[Any]:
    """Simulate T observations of (Index_return, H, E, M)."""
    means = np.array([mu_idx, mu_H, mu_E, mu_M])
    return np.random.multivariate_normal(means, cov, size=T)
