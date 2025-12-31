from __future__ import annotations

from typing import Dict, Tuple, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .agents.registry import build_from_config
from .config import ModelConfig
from .random import spawn_agent_rngs, spawn_rngs
from .sim.covariance import build_cov_matrix
from .sim.metrics import summary_table
from .sim.paths import draw_financing_series, draw_joint_returns
from .simulations import simulate_agents
from .validators import select_vol_regime_sigma

Array: TypeAlias = NDArray[np.float64]


class SimulatorOrchestrator:
    """Run Monte Carlo simulations and compute summary metrics."""

    def __init__(self, cfg: ModelConfig, idx_series: pd.Series) -> None:
        self.cfg = cfg
        self.idx_series = idx_series

    def run(self, seed: int | None = None) -> Tuple[Dict[str, Array], pd.DataFrame]:
        """Execute simulations and return per-agent returns and summary table."""

        mu_idx = float(self.idx_series.mean())
        idx_sigma, _, _ = select_vol_regime_sigma(
            self.idx_series,
            regime=self.cfg.vol_regime,
            window=self.cfg.vol_regime_window,
        )
        n_samples = int(len(self.idx_series))

        _ = build_cov_matrix(
            self.cfg.rho_idx_H,
            self.cfg.rho_idx_E,
            self.cfg.rho_idx_M,
            self.cfg.rho_H_E,
            self.cfg.rho_H_M,
            self.cfg.rho_E_M,
            idx_sigma,
            self.cfg.sigma_H,
            self.cfg.sigma_E,
            self.cfg.sigma_M,
            covariance_shrinkage=self.cfg.covariance_shrinkage,
            n_samples=n_samples,
        )

        rng_returns = spawn_rngs(seed, 1)[0]
        params = {
            "mu_idx_month": mu_idx / 12,
            "default_mu_H": self.cfg.mu_H / 12,
            "default_mu_E": self.cfg.mu_E / 12,
            "default_mu_M": self.cfg.mu_M / 12,
            "idx_sigma_month": idx_sigma / 12,
            "default_sigma_H": self.cfg.sigma_H / 12,
            "default_sigma_E": self.cfg.sigma_E / 12,
            "default_sigma_M": self.cfg.sigma_M / 12,
            "rho_idx_H": self.cfg.rho_idx_H,
            "rho_idx_E": self.cfg.rho_idx_E,
            "rho_idx_M": self.cfg.rho_idx_M,
            "rho_H_E": self.cfg.rho_H_E,
            "rho_H_M": self.cfg.rho_H_M,
            "rho_E_M": self.cfg.rho_E_M,
            "return_distribution": self.cfg.return_distribution,
            "return_t_df": self.cfg.return_t_df,
            "return_copula": self.cfg.return_copula,
            "return_distribution_idx": self.cfg.return_distribution_idx,
            "return_distribution_H": self.cfg.return_distribution_H,
            "return_distribution_E": self.cfg.return_distribution_E,
            "return_distribution_M": self.cfg.return_distribution_M,
        }

        r_beta, r_H, r_E, r_M = draw_joint_returns(
            n_months=self.cfg.N_MONTHS,
            n_sim=self.cfg.N_SIMULATIONS,
            params=params,
            rng=rng_returns,
        )

        fin_params = {
            "internal_financing_mean_month": self.cfg.internal_financing_mean_month,
            "internal_financing_sigma_month": self.cfg.internal_financing_sigma_month,
            "internal_spike_prob": self.cfg.internal_spike_prob,
            "internal_spike_factor": self.cfg.internal_spike_factor,
            "ext_pa_financing_mean_month": self.cfg.ext_pa_financing_mean_month,
            "ext_pa_financing_sigma_month": self.cfg.ext_pa_financing_sigma_month,
            "ext_pa_spike_prob": self.cfg.ext_pa_spike_prob,
            "ext_pa_spike_factor": self.cfg.ext_pa_spike_factor,
            "act_ext_financing_mean_month": self.cfg.act_ext_financing_mean_month,
            "act_ext_financing_sigma_month": self.cfg.act_ext_financing_sigma_month,
            "act_ext_spike_prob": self.cfg.act_ext_spike_prob,
            "act_ext_spike_factor": self.cfg.act_ext_spike_factor,
        }
        fin_rngs = spawn_agent_rngs(seed, ["internal", "external_pa", "active_ext"])
        f_int, f_ext, f_act = draw_financing_series(
            n_months=self.cfg.N_MONTHS,
            n_sim=self.cfg.N_SIMULATIONS,
            params=fin_params,
            rngs=fin_rngs,
        )

        agents = build_from_config(self.cfg)
        returns = simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act)
        summary = summary_table(returns, benchmark="Base")
        return returns, summary
