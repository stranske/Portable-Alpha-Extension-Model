from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .agents.registry import build_from_config
from .config import CANONICAL_RETURN_UNIT, ModelConfig
from .random import spawn_agent_rngs, spawn_rngs
from .sim.covariance import build_cov_matrix
from .sim.metrics import summary_table
from .sim.params import build_simulation_params
from .sim.paths import draw_financing_series, draw_joint_returns
from .simulations import simulate_agents
from .types import ArrayLike
from .units import normalize_index_series
from .validators import select_vol_regime_sigma


def _cov_to_corr_and_sigma(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(sigma, sigma)
    corr = np.divide(cov, denom, out=np.eye(cov.shape[0]), where=denom != 0.0)
    return sigma, corr


class SimulatorOrchestrator:
    """Run Monte Carlo simulations using PSD-corrected covariance inputs.

    Builds a covariance matrix from configured sigmas/correlations, applies
    PSD projection when needed, then derives implied volatilities and
    correlations from the corrected matrix for return draws.
    """

    def __init__(self, cfg: ModelConfig, idx_series: pd.Series) -> None:
        self.cfg = cfg
        self.idx_series = idx_series

    def run(self, seed: int | None = None) -> Tuple[Dict[str, ArrayLike], pd.DataFrame]:
        """Execute simulations and return per-agent returns and summary table.

        Uses the PSD-corrected covariance matrix to derive implied
        volatilities and correlations before drawing joint returns.
        """

        idx_series = normalize_index_series(
            self.idx_series,
            getattr(self.cfg, "input_return_unit", CANONICAL_RETURN_UNIT),
        )
        mu_idx = float(idx_series.mean())
        idx_sigma, _, _ = select_vol_regime_sigma(
            idx_series,
            regime=self.cfg.vol_regime,
            window=self.cfg.vol_regime_window,
        )
        n_samples = int(len(idx_series))

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
        return_overrides = None
        idx_sigma_use = idx_sigma
        if self.cfg.covariance_shrinkage != "none":
            sigma_vec, corr_mat = _cov_to_corr_and_sigma(cov)
            idx_sigma_use = float(sigma_vec[0]) / 12
            sigma_h_cov = float(sigma_vec[1])
            sigma_e_cov = float(sigma_vec[2])
            sigma_m_cov = float(sigma_vec[3])
            return_overrides = {
                "default_sigma_H": sigma_h_cov / 12,
                "default_sigma_E": sigma_e_cov / 12,
                "default_sigma_M": sigma_m_cov / 12,
                "rho_idx_H": float(corr_mat[0, 1]),
                "rho_idx_E": float(corr_mat[0, 2]),
                "rho_idx_M": float(corr_mat[0, 3]),
                "rho_H_E": float(corr_mat[1, 2]),
                "rho_H_M": float(corr_mat[1, 3]),
                "rho_E_M": float(corr_mat[2, 3]),
            }

        rng_returns = spawn_rngs(seed, 1)[0]
        params = build_simulation_params(
            self.cfg,
            mu_idx=mu_idx,
            idx_sigma=idx_sigma_use,
            return_overrides=return_overrides,
        )

        r_beta, r_H, r_E, r_M = draw_joint_returns(
            n_months=self.cfg.N_MONTHS,
            n_sim=self.cfg.N_SIMULATIONS,
            params=params,
            rng=rng_returns,
        )

        fin_rngs = spawn_agent_rngs(seed, ["internal", "external_pa", "active_ext"])
        f_int, f_ext, f_act = draw_financing_series(
            n_months=self.cfg.N_MONTHS,
            n_sim=self.cfg.N_SIMULATIONS,
            params=params,
            rngs=fin_rngs,
        )

        agents = build_from_config(self.cfg)
        returns = simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act)
        summary = summary_table(returns, benchmark="Base")
        return returns, summary
