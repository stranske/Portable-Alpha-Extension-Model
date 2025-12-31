from __future__ import annotations

from typing import Dict, Tuple

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


class SimulatorOrchestrator:
    """Run Monte Carlo simulations and compute summary metrics."""

    def __init__(self, cfg: ModelConfig, idx_series: pd.Series) -> None:
        self.cfg = cfg
        self.idx_series = idx_series

    def run(self, seed: int | None = None) -> Tuple[Dict[str, ArrayLike], pd.DataFrame]:
        """Execute simulations and return per-agent returns and summary table."""

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
        params = build_simulation_params(self.cfg, mu_idx=mu_idx, idx_sigma=idx_sigma)

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
