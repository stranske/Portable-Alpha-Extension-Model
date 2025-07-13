from __future__ import annotations

from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd

from .agents.registry import build_from_config
from .config import ModelConfig
from .sim import draw_financing_series, draw_joint_returns
from .sim.covariance import build_cov_matrix
from .sim.metrics import summary_table
from .simulations import simulate_agents


def generate_parameter_combinations(cfg: ModelConfig) -> Iterator[Dict[str, Any]]:
    """Generate parameter combinations based on ``analysis_mode``."""
    if cfg.analysis_mode == "capital":
        for ext_pct in np.arange(
            0,
            cfg.max_external_combined_pct + cfg.external_step_size_pct,
            cfg.external_step_size_pct,
        ):
            for act_pct in np.arange(
                0,
                ext_pct + cfg.external_step_size_pct,
                cfg.external_step_size_pct,
            ):
                ext_pa_pct = ext_pct - act_pct
                internal_pct = 100 - ext_pct
                yield {
                    "external_pa_capital": (ext_pa_pct / 100) * cfg.total_fund_capital,
                    "active_ext_capital": (act_pct / 100) * cfg.total_fund_capital,
                    "internal_pa_capital": (internal_pct / 100)
                    * cfg.total_fund_capital,
                }
    elif cfg.analysis_mode == "returns":
        for mu_H in np.arange(
            cfg.in_house_return_min_pct,
            cfg.in_house_return_max_pct + cfg.in_house_return_step_pct,
            cfg.in_house_return_step_pct,
        ):
            for sigma_H in np.arange(
                cfg.in_house_vol_min_pct,
                cfg.in_house_vol_max_pct + cfg.in_house_vol_step_pct,
                cfg.in_house_vol_step_pct,
            ):
                for mu_E in np.arange(
                    cfg.alpha_ext_return_min_pct,
                    cfg.alpha_ext_return_max_pct + cfg.alpha_ext_return_step_pct,
                    cfg.alpha_ext_return_step_pct,
                ):
                    for sigma_E in np.arange(
                        cfg.alpha_ext_vol_min_pct,
                        cfg.alpha_ext_vol_max_pct + cfg.alpha_ext_vol_step_pct,
                        cfg.alpha_ext_vol_step_pct,
                    ):
                        yield {
                            "mu_H": mu_H / 100,
                            "sigma_H": sigma_H / 100,
                            "mu_E": mu_E / 100,
                            "sigma_E": sigma_E / 100,
                        }
    elif cfg.analysis_mode == "alpha_shares":
        for theta_extpa in np.arange(
            cfg.external_pa_alpha_min_pct,
            cfg.external_pa_alpha_max_pct + cfg.external_pa_alpha_step_pct,
            cfg.external_pa_alpha_step_pct,
        ):
            for active_share in np.arange(
                cfg.active_share_min_pct,
                cfg.active_share_max_pct + cfg.active_share_step_pct,
                cfg.active_share_step_pct,
            ):
                yield {
                    "theta_extpa": theta_extpa / 100,
                    "active_share": active_share,
                }
    elif cfg.analysis_mode == "vol_mult":
        for sd_mult in np.arange(
            cfg.sd_multiple_min,
            cfg.sd_multiple_max + cfg.sd_multiple_step,
            cfg.sd_multiple_step,
        ):
            yield {
                "sigma_H": cfg.sigma_H * sd_mult,
                "sigma_E": cfg.sigma_E * sd_mult,
                "sigma_M": cfg.sigma_M * sd_mult,
            }
    else:
        raise ValueError(f"Unsupported analysis mode: {cfg.analysis_mode}")


def run_parameter_sweep(
    cfg: ModelConfig,
    index_series: pd.Series,
    rng_returns: np.random.Generator,
    fin_rngs: Dict[str, np.random.Generator],
) -> List[Dict[str, Any]]:
    """Run the parameter sweep and collect results."""
    results: List[Dict[str, Any]] = []

    mu_idx = float(index_series.mean())
    idx_sigma = float(index_series.std(ddof=1))

    for i, overrides in enumerate(generate_parameter_combinations(cfg)):
        mod_cfg = cfg.model_copy(update=overrides)

        build_cov_matrix(
            mod_cfg.rho_idx_H,
            mod_cfg.rho_idx_E,
            mod_cfg.rho_idx_M,
            mod_cfg.rho_H_E,
            mod_cfg.rho_H_M,
            mod_cfg.rho_E_M,
            idx_sigma,
            mod_cfg.sigma_H,
            mod_cfg.sigma_E,
            mod_cfg.sigma_M,
        )
        params = {
            "mu_idx_month": mu_idx / 12,
            "default_mu_H": mod_cfg.mu_H / 12,
            "default_mu_E": mod_cfg.mu_E / 12,
            "default_mu_M": mod_cfg.mu_M / 12,
            "idx_sigma_month": idx_sigma / 12,
            "default_sigma_H": mod_cfg.sigma_H / 12,
            "default_sigma_E": mod_cfg.sigma_E / 12,
            "default_sigma_M": mod_cfg.sigma_M / 12,
            "rho_idx_H": mod_cfg.rho_idx_H,
            "rho_idx_E": mod_cfg.rho_idx_E,
            "rho_idx_M": mod_cfg.rho_idx_M,
            "rho_H_E": mod_cfg.rho_H_E,
            "rho_H_M": mod_cfg.rho_H_M,
            "rho_E_M": mod_cfg.rho_E_M,
            "internal_financing_mean_month": mod_cfg.internal_financing_mean_month,
            "internal_financing_sigma_month": mod_cfg.internal_financing_sigma_month,
            "internal_spike_prob": mod_cfg.internal_spike_prob,
            "internal_spike_factor": mod_cfg.internal_spike_factor,
            "ext_pa_financing_mean_month": mod_cfg.ext_pa_financing_mean_month,
            "ext_pa_financing_sigma_month": mod_cfg.ext_pa_financing_sigma_month,
            "ext_pa_spike_prob": mod_cfg.ext_pa_spike_prob,
            "ext_pa_spike_factor": mod_cfg.ext_pa_spike_factor,
            "act_ext_financing_mean_month": mod_cfg.act_ext_financing_mean_month,
            "act_ext_financing_sigma_month": mod_cfg.act_ext_financing_sigma_month,
            "act_ext_spike_prob": mod_cfg.act_ext_spike_prob,
            "act_ext_spike_factor": mod_cfg.act_ext_spike_factor,
        }

        r_beta, r_H, r_E, r_M = draw_joint_returns(
            n_months=mod_cfg.N_MONTHS,
            n_sim=mod_cfg.N_SIMULATIONS,
            params=params,
            rng=rng_returns,
        )
        f_int, f_ext, f_act = draw_financing_series(
            n_months=mod_cfg.N_MONTHS,
            n_sim=mod_cfg.N_SIMULATIONS,
            params=params,
            rngs={"financing": fin_rngs["financing"]} if fin_rngs else None,
        )

        agents = build_from_config(mod_cfg)
        returns = simulate_agents(
            agents,
            r_beta,
            r_H,
            r_E,
            r_M,
            f_int,
            f_ext,
            f_act,
        )

        summary = summary_table(returns, benchmark="Base")
        results.append(
            {
                "combination_id": i,
                "parameters": overrides,
                "summary": summary,
            }
        )

    return results
