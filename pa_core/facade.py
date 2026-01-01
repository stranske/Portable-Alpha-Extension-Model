"""Facade types for CLI and programmatic run entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypedDict

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import numpy as np
    import pandas as pd

    from .config import ModelConfig
from .types import ArrayLike


@dataclass(slots=True)
class RunArtifacts:
    """Standardized outputs from a single simulation run."""

    config: "ModelConfig"
    index_series: "pd.Series"
    returns: dict[str, ArrayLike]
    summary: "pd.DataFrame"
    inputs: dict[str, Any]
    raw_returns: dict[str, "pd.DataFrame"]
    stress_delta: "pd.DataFrame | None" = None
    base_summary: "pd.DataFrame | None" = None
    manifest: Mapping[str, Any] | None = None


class SweepResult(TypedDict):
    """Standardized outputs from a single sweep combination."""

    combination_id: int
    parameters: Mapping[str, Any]
    summary: "pd.DataFrame"


@dataclass(slots=True)
class SweepArtifacts:
    """Standardized outputs from a parameter sweep run."""

    config: "ModelConfig"
    index_series: "pd.Series"
    results: Sequence[SweepResult]
    summary: "pd.DataFrame"
    inputs: dict[str, Any]
    manifest: Mapping[str, Any] | None = None


@dataclass(slots=True)
class RunOptions:
    """Configuration overrides for programmatic run entrypoints."""

    seed: int | None = None
    backend: str | None = None
    config_overrides: Mapping[str, Any] | None = None


def _cov_to_corr_and_sigma(cov: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(sigma, sigma)
    corr = np.divide(cov, denom, out=np.eye(cov.shape[0]), where=denom != 0.0)
    return sigma, corr


def run_single(
    config: "ModelConfig",
    index_series: "pd.Series",
    options: RunOptions | None = None,
) -> RunArtifacts:
    """Run a single simulation with optional overrides and return artifacts."""

    import pandas as pd

    from .agents.registry import build_from_config
    from .backend import resolve_and_set_backend
    from .random import spawn_agent_rngs, spawn_rngs
    from .sim import draw_financing_series, draw_joint_returns
    from .sim.covariance import build_cov_matrix
    from .sim.metrics import summary_table
    from .sim.params import build_simulation_params
    from .simulations import simulate_agents
    from .validators import select_vol_regime_sigma

    run_options = options or RunOptions()
    run_cfg = (
        config.model_copy(update=dict(run_options.config_overrides))
        if run_options.config_overrides
        else config
    )
    resolve_and_set_backend(run_options.backend, run_cfg)

    idx_series = index_series
    if isinstance(idx_series, pd.DataFrame):
        idx_series = idx_series.squeeze()
        if not isinstance(idx_series, pd.Series):
            raise ValueError("Index data must be convertible to pandas Series")
    elif not isinstance(idx_series, pd.Series):
        raise ValueError("Index data must be a pandas Series")

    mu_idx = float(idx_series.mean())
    idx_sigma, _, _ = select_vol_regime_sigma(
        idx_series,
        regime=run_cfg.vol_regime,
        window=run_cfg.vol_regime_window,
    )
    n_samples = int(len(idx_series))

    sigma_h = float(run_cfg.sigma_H)
    sigma_e = float(run_cfg.sigma_E)
    sigma_m = float(run_cfg.sigma_M)

    cov = build_cov_matrix(
        run_cfg.rho_idx_H,
        run_cfg.rho_idx_E,
        run_cfg.rho_idx_M,
        run_cfg.rho_H_E,
        run_cfg.rho_H_M,
        run_cfg.rho_E_M,
        idx_sigma,
        sigma_h,
        sigma_e,
        sigma_m,
        covariance_shrinkage=run_cfg.covariance_shrinkage,
        n_samples=n_samples,
    )
    sigma_vec, corr_mat = _cov_to_corr_and_sigma(cov)

    params = build_simulation_params(
        run_cfg,
        mu_idx=mu_idx,
        idx_sigma=float(sigma_vec[0]),
        return_overrides={
            "default_sigma_H": float(sigma_vec[1]),
            "default_sigma_E": float(sigma_vec[2]),
            "default_sigma_M": float(sigma_vec[3]),
            "rho_idx_H": float(corr_mat[0, 1]),
            "rho_idx_E": float(corr_mat[0, 2]),
            "rho_idx_M": float(corr_mat[0, 3]),
            "rho_H_E": float(corr_mat[1, 2]),
            "rho_H_M": float(corr_mat[1, 3]),
            "rho_E_M": float(corr_mat[2, 3]),
        },
    )

    rng_returns = spawn_rngs(run_options.seed, 1)[0]
    r_beta, r_H, r_E, r_M = draw_joint_returns(
        n_months=run_cfg.N_MONTHS,
        n_sim=run_cfg.N_SIMULATIONS,
        params=params,
        rng=rng_returns,
    )
    fin_rngs = spawn_agent_rngs(run_options.seed, ["internal", "external_pa", "active_ext"])
    f_int, f_ext, f_act = draw_financing_series(
        n_months=run_cfg.N_MONTHS,
        n_sim=run_cfg.N_SIMULATIONS,
        params=params,
        rngs=fin_rngs,
    )

    agents = build_from_config(run_cfg)
    returns = simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act)
    summary = summary_table(returns, benchmark="Base")
    raw_returns = {name: pd.DataFrame(data) for name, data in returns.items()}
    inputs = run_cfg.model_dump()

    return RunArtifacts(
        config=run_cfg,
        index_series=idx_series,
        returns=returns,
        summary=summary,
        inputs=inputs,
        raw_returns=raw_returns,
    )


def run_sweep(
    config: "ModelConfig",
    index_series: "pd.Series",
    sweep_params: Mapping[str, Any] | None = None,
    options: RunOptions | None = None,
) -> SweepArtifacts:
    """Run a parameter sweep with optional overrides and return artifacts."""

    import pandas as pd

    from .backend import resolve_and_set_backend
    from .random import spawn_agent_rngs, spawn_rngs
    from .sweep import run_parameter_sweep, sweep_results_to_dataframe

    run_options = options or RunOptions()
    updates: dict[str, Any] = {}
    if run_options.config_overrides:
        updates.update(run_options.config_overrides)
    if sweep_params:
        updates.update(sweep_params)
    run_cfg = config.model_copy(update=updates) if updates else config
    resolve_and_set_backend(run_options.backend, run_cfg)

    idx_series = index_series
    if isinstance(idx_series, pd.DataFrame):
        idx_series = idx_series.squeeze()
        if not isinstance(idx_series, pd.Series):
            raise ValueError("Index data must be convertible to pandas Series")
    elif not isinstance(idx_series, pd.Series):
        raise ValueError("Index data must be a pandas Series")

    rng_returns = spawn_rngs(run_options.seed, 1)[0]
    fin_rngs = spawn_agent_rngs(run_options.seed, ["internal", "external_pa", "active_ext"])
    results = run_parameter_sweep(
        run_cfg,
        idx_series,
        rng_returns,
        fin_rngs,
        seed=run_options.seed,
    )
    summary = sweep_results_to_dataframe(results)
    inputs = run_cfg.model_dump()

    return SweepArtifacts(
        config=run_cfg,
        index_series=idx_series,
        results=results,
        summary=summary,
        inputs=inputs,
    )
