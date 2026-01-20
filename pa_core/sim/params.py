from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..config import ModelConfig
from ..units import (
    convert_mean,
    convert_volatility,
    get_index_series_unit,
    normalize_return_inputs,
)

CANONICAL_PARAMS_MARKER = "__pa_sim_params__"
CANONICAL_PARAMS_VERSION = "build_simulation_params_v1"


def build_return_params(cfg: ModelConfig, *, mu_idx: float, idx_sigma: float) -> Dict[str, Any]:
    """Return draw parameters shared across CLI, sweep, and orchestrator."""
    return_inputs = normalize_return_inputs(cfg)
    index_unit = get_index_series_unit()
    return {
        "mu_idx_month": convert_mean(mu_idx, from_unit=index_unit, to_unit="monthly"),
        "default_mu_H": return_inputs["mu_H"],
        "default_mu_E": return_inputs["mu_E"],
        "default_mu_M": return_inputs["mu_M"],
        "idx_sigma_month": convert_volatility(idx_sigma, from_unit=index_unit, to_unit="monthly"),
        "default_sigma_H": return_inputs["sigma_H"],
        "default_sigma_E": return_inputs["sigma_E"],
        "default_sigma_M": return_inputs["sigma_M"],
        "rho_idx_H": cfg.rho_idx_H,
        "rho_idx_E": cfg.rho_idx_E,
        "rho_idx_M": cfg.rho_idx_M,
        "rho_H_E": cfg.rho_H_E,
        "rho_H_M": cfg.rho_H_M,
        "rho_E_M": cfg.rho_E_M,
        "correlation_repair_mode": cfg.correlation_repair_mode,
        "correlation_repair_shrinkage": cfg.correlation_repair_shrinkage,
        "correlation_repair_max_abs_delta": cfg.correlation_repair_max_abs_delta,
        "return_distribution": cfg.return_distribution,
        "return_t_df": cfg.return_t_df,
        "return_copula": cfg.return_copula,
        "return_distribution_idx": cfg.return_distribution_idx,
        "return_distribution_H": cfg.return_distribution_H,
        "return_distribution_E": cfg.return_distribution_E,
        "return_distribution_M": cfg.return_distribution_M,
    }


def build_financing_params(cfg: ModelConfig) -> Dict[str, Any]:
    """Return financing draw parameters shared across CLI, sweep, and orchestrator."""
    return {
        "internal_financing_mean_month": cfg.internal_financing_mean_month,
        "internal_financing_sigma_month": cfg.internal_financing_sigma_month,
        "internal_spike_prob": cfg.internal_spike_prob,
        "internal_spike_factor": cfg.internal_spike_factor,
        "ext_pa_financing_mean_month": cfg.ext_pa_financing_mean_month,
        "ext_pa_financing_sigma_month": cfg.ext_pa_financing_sigma_month,
        "ext_pa_spike_prob": cfg.ext_pa_spike_prob,
        "ext_pa_spike_factor": cfg.ext_pa_spike_factor,
        "act_ext_financing_mean_month": cfg.act_ext_financing_mean_month,
        "act_ext_financing_sigma_month": cfg.act_ext_financing_sigma_month,
        "act_ext_spike_prob": cfg.act_ext_spike_prob,
        "act_ext_spike_factor": cfg.act_ext_spike_factor,
    }


def resolve_covariance_inputs(
    cov: Any,
    *,
    idx_sigma: float,
    sigma_h: float,
    sigma_e: float,
    sigma_m: float,
    rho_idx_H: float,
    rho_idx_E: float,
    rho_idx_M: float,
    rho_H_E: float,
    rho_H_M: float,
    rho_E_M: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sigma_vec, corr_mat) from covariance, with config fallbacks."""
    if isinstance(cov, np.ndarray) and cov.ndim == 2:
        sigma_vec = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        denom = np.outer(sigma_vec, sigma_vec)
        corr_mat = np.divide(cov, denom, out=np.eye(cov.shape[0]), where=denom != 0.0)
    else:
        sigma_vec = np.array([idx_sigma, sigma_h, sigma_e, sigma_m], dtype=float)
        corr_mat = np.array(
            [
                [1.0, rho_idx_H, rho_idx_E, rho_idx_M],
                [rho_idx_H, 1.0, rho_H_E, rho_H_M],
                [rho_idx_E, rho_H_E, 1.0, rho_E_M],
                [rho_idx_M, rho_H_M, rho_E_M, 1.0],
            ],
            dtype=float,
        )
    return sigma_vec, corr_mat


def build_covariance_return_overrides(
    sigma_vec: np.ndarray, corr_mat: np.ndarray
) -> Dict[str, float]:
    """Return overrides derived from covariance-implied sigma/correlation."""
    return {
        "default_sigma_H": float(sigma_vec[1]),
        "default_sigma_E": float(sigma_vec[2]),
        "default_sigma_M": float(sigma_vec[3]),
        "rho_idx_H": float(corr_mat[0, 1]),
        "rho_idx_E": float(corr_mat[0, 2]),
        "rho_idx_M": float(corr_mat[0, 3]),
        "rho_H_E": float(corr_mat[1, 2]),
        "rho_H_M": float(corr_mat[1, 3]),
        "rho_E_M": float(corr_mat[2, 3]),
    }


def build_simulation_params(
    cfg: ModelConfig,
    *,
    mu_idx: float,
    idx_sigma: float,
    return_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine return and financing parameters for simulation draws."""
    params = build_return_params(cfg, mu_idx=mu_idx, idx_sigma=idx_sigma)
    if return_overrides:
        params.update(return_overrides)
    params.update(build_financing_params(cfg))
    params[CANONICAL_PARAMS_MARKER] = CANONICAL_PARAMS_VERSION
    return params


def build_params(
    cfg: ModelConfig,
    *,
    mu_idx: float,
    idx_sigma: float,
    return_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical build alias for simulation params."""
    return build_simulation_params(
        cfg,
        mu_idx=mu_idx,
        idx_sigma=idx_sigma,
        return_overrides=return_overrides,
    )
