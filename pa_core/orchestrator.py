from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd

from .config import ModelConfig
from .facade import RunOptions, run_single, run_sweep
from .types import ArrayLike, SweepResult
from .units import get_index_series_unit, normalize_index_series


class SimulatorOrchestrator:
    """Run Monte Carlo simulations using PSD-corrected covariance inputs.

    Builds a covariance matrix from configured sigmas/correlations, applies
    PSD projection when needed, then derives implied volatilities and
    correlations from the corrected matrix for return draws.
    """

    def __init__(self, cfg: ModelConfig, idx_series: pd.Series) -> None:
        self.cfg = cfg
        self.idx_series = normalize_index_series(pd.Series(idx_series), get_index_series_unit())

    def run(self, seed: int | None = None) -> Tuple[Dict[str, ArrayLike], pd.DataFrame]:
        """Execute simulations and return per-agent returns and summary table.

        Uses the PSD-corrected covariance matrix to derive implied
        volatilities and correlations before drawing joint returns. Summary
        table metrics (terminal_AnnReturn/monthly_AnnVol/monthly_TE) are annualised
        from monthly returns.
        """

        artifacts = run_single(self.cfg, self.idx_series, RunOptions(seed=seed))
        return artifacts.returns, artifacts.summary

    def run_sweep(
        self,
        sweep_params: Mapping[str, object] | None = None,
        seed: int | None = None,
    ) -> Tuple[Sequence[SweepResult], pd.DataFrame]:
        """Execute a parameter sweep and return results plus consolidated summary."""
        artifacts = run_sweep(
            self.cfg,
            self.idx_series,
            sweep_params,
            RunOptions(seed=seed),
        )
        return artifacts.results, artifacts.summary
