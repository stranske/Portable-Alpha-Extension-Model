from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .agents.registry import build_from_config
from .config import ModelConfig
from .random import spawn_agent_rngs, spawn_rngs
from .sim.covariance import build_cov_matrix
from .sim.metrics import summary_table
from .sim.params import (
    build_covariance_return_overrides,
    build_params,
    resolve_covariance_inputs,
)
from .sim.paths import draw_financing_series, draw_joint_returns
from .simulations import simulate_agents
from .types import ArrayLike
from .units import get_index_series_unit, normalize_index_series, normalize_return_inputs
from .validators import select_vol_regime_sigma


class SimulatorOrchestrator:
    """Run Monte Carlo simulations using PSD-corrected covariance inputs.

    Builds a covariance matrix from configured sigmas/correlations, applies
    PSD projection when needed, then derives implied volatilities and
    correlations from the corrected matrix for return draws.
    """

    def __init__(self, cfg: ModelConfig, idx_series: pd.Series) -> None:
        self.cfg = cfg
        self.idx_series = normalize_index_series(pd.Series(idx_series), get_index_series_unit())

    def draw_streams(self, seed: int | None = None) -> Tuple[ArrayLike, ...]:
        """Draw Monte Carlo return and financing streams for the configured model."""
        mu_idx = float(self.idx_series.mean())
        idx_sigma, _, _ = select_vol_regime_sigma(
            self.idx_series,
            regime=self.cfg.vol_regime,
            window=self.cfg.vol_regime_window,
        )
        n_samples = int(len(self.idx_series))

        return_inputs = normalize_return_inputs(self.cfg)
        sigma_h = float(return_inputs["sigma_H"])
        sigma_e = float(return_inputs["sigma_E"])
        sigma_m = float(return_inputs["sigma_M"])

        cov = build_cov_matrix(
            self.cfg.rho_idx_H,
            self.cfg.rho_idx_E,
            self.cfg.rho_idx_M,
            self.cfg.rho_H_E,
            self.cfg.rho_H_M,
            self.cfg.rho_E_M,
            idx_sigma,
            sigma_h,
            sigma_e,
            sigma_m,
            covariance_shrinkage=self.cfg.covariance_shrinkage,
            n_samples=n_samples,
        )
        sigma_vec, corr_mat = resolve_covariance_inputs(
            cov,
            idx_sigma=idx_sigma,
            sigma_h=sigma_h,
            sigma_e=sigma_e,
            sigma_m=sigma_m,
            rho_idx_H=self.cfg.rho_idx_H,
            rho_idx_E=self.cfg.rho_idx_E,
            rho_idx_M=self.cfg.rho_idx_M,
            rho_H_E=self.cfg.rho_H_E,
            rho_H_M=self.cfg.rho_H_M,
            rho_E_M=self.cfg.rho_E_M,
        )

        rng_returns = spawn_rngs(seed, 1)[0]
        params = build_params(
            self.cfg,
            mu_idx=mu_idx,
            idx_sigma=float(sigma_vec[0]),
            return_overrides=build_covariance_return_overrides(sigma_vec, corr_mat),
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
        return r_beta, r_H, r_E, r_M, f_int, f_ext, f_act

    def run(self, seed: int | None = None) -> Tuple[Dict[str, ArrayLike], pd.DataFrame]:
        """Execute simulations and return per-agent returns and summary table.

        Uses the PSD-corrected covariance matrix to derive implied
        volatilities and correlations before drawing joint returns. Summary
        table metrics (AnnReturn/AnnVol/TE) are annualised from monthly returns.
        """

        r_beta, r_H, r_E, r_M, f_int, f_ext, f_act = self.draw_streams(seed=seed)
        agents = build_from_config(self.cfg)
        returns = simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act)
        summary = summary_table(returns, benchmark="Base")
        return returns, summary
