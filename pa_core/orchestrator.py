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
from .sim.paths import draw_financing_series, prepare_mc_universe
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

        cov = build_cov_matrix(
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
        universe = prepare_mc_universe(
            N_SIMULATIONS=self.cfg.N_SIMULATIONS,
            N_MONTHS=self.cfg.N_MONTHS,
            mu_idx=mu_idx,
            mu_H=self.cfg.mu_H,
            mu_E=self.cfg.mu_E,
            mu_M=self.cfg.mu_M,
            cov_mat=cov,
            rng=rng_returns,
        )
        r_beta = universe[:, :, 0]
        r_H = universe[:, :, 1]
        r_E = universe[:, :, 2]
        r_M = universe[:, :, 3]

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
