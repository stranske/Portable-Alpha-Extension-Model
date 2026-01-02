from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import ModelConfig

CANONICAL_PARAMS_MARKER = "__pa_sim_params__"
CANONICAL_PARAMS_VERSION = "build_simulation_params_v1"


def build_return_params(cfg: ModelConfig, *, mu_idx: float, idx_sigma: float) -> Dict[str, Any]:
    """Return draw parameters shared across CLI, sweep, and orchestrator."""
    return {
        "mu_idx_month": mu_idx,
        "default_mu_H": cfg.mu_H,
        "default_mu_E": cfg.mu_E,
        "default_mu_M": cfg.mu_M,
        "idx_sigma_month": idx_sigma,
        "default_sigma_H": cfg.sigma_H,
        "default_sigma_E": cfg.sigma_E,
        "default_sigma_M": cfg.sigma_M,
        "rho_idx_H": cfg.rho_idx_H,
        "rho_idx_E": cfg.rho_idx_E,
        "rho_idx_M": cfg.rho_idx_M,
        "rho_H_E": cfg.rho_H_E,
        "rho_H_M": cfg.rho_H_M,
        "rho_E_M": cfg.rho_E_M,
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
