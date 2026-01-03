"""Facade types for CLI and programmatic run entrypoints.

This module provides a clean programmatic API for running simulations,
decoupling business logic from CLI argument parsing. It enables:

- Direct programmatic access to simulations without CLI
- Standardized artifact types for consistent outputs
- Type-safe configuration with clear interfaces

Example Usage::

    from pa_core.config import load_config
    from pa_core.facade import run_single, export, RunOptions

    config = load_config("config.yml")
    index_series = pd.read_csv("index.csv")["returns"]

    # Run simulation
    artifacts = run_single(config, index_series, RunOptions(seed=42))

    # Export results
    export(artifacts, "results.xlsx")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import pandas as pd

    from .config import ModelConfig
from .types import ArrayLike, SweepResult


@dataclass(slots=True)
class RunArtifacts:
    """Standardized outputs from a single simulation run.

    Attributes:
        config: The resolved configuration used for the simulation.
        index_series: The benchmark index returns series.
        returns: Dictionary mapping agent names to return arrays.
        summary: DataFrame with summary statistics per agent.
        inputs: Dictionary of input parameters used.
        raw_returns: Dictionary mapping agent names to DataFrames of returns.
        stress_delta: Optional DataFrame with stress scenario deltas.
        base_summary: Optional DataFrame with base scenario summary (for stress comparisons).
        manifest: Optional metadata dictionary for reproducibility tracking.
    """

    config: "ModelConfig"
    index_series: "pd.Series"
    returns: dict[str, ArrayLike]
    summary: "pd.DataFrame"
    inputs: dict[str, Any]
    raw_returns: dict[str, "pd.DataFrame"]
    stress_delta: "pd.DataFrame | None" = None
    base_summary: "pd.DataFrame | None" = None
    manifest: Mapping[str, Any] | None = None


@dataclass(slots=True)
class SweepArtifacts:
    """Standardized outputs from a parameter sweep run.

    Attributes:
        config: The resolved configuration used for the sweep.
        index_series: The benchmark index returns series.
        results: Sequence of SweepResult dictionaries, one per parameter combination.
        summary: Consolidated DataFrame with all sweep results.
        inputs: Dictionary of input parameters used.
        manifest: Optional metadata dictionary for reproducibility tracking.
    """

    config: "ModelConfig"
    index_series: "pd.Series"
    results: Sequence[SweepResult]
    summary: "pd.DataFrame"
    inputs: dict[str, Any]
    manifest: Mapping[str, Any] | None = None


@dataclass(slots=True)
class RunOptions:
    """Configuration overrides for programmatic run entrypoints.

    Attributes:
        seed: Random seed for reproducible simulations. If None, uses non-deterministic RNG.
        backend: Computation backend selection (currently only "numpy" supported).
        config_overrides: Dictionary of config parameter overrides to apply.
        legacy_agent_rng: Use order-dependent agent RNG streams for backward compatibility.
    """

    seed: int | None = None
    backend: str | None = None
    config_overrides: Mapping[str, Any] | None = None
    legacy_agent_rng: bool = False


@dataclass(slots=True)
class ExportOptions:
    """Configuration options for exporting artifacts.

    Attributes:
        pivot: If True, write all returns in a single long-format sheet.
        include_sensitivity: If True, include sensitivity analysis in exports.
        include_charts: If True, embed charts in Excel output.
        alt_text: Alt text for accessibility in exported charts.
    """

    pivot: bool = False
    include_sensitivity: bool = False
    include_charts: bool = True
    alt_text: str | None = None


def run_single(
    config: "ModelConfig",
    index_series: "pd.Series",
    options: RunOptions | None = None,
) -> RunArtifacts:
    """Run a single simulation with optional overrides and return artifacts.

    This is the primary programmatic entrypoint for running a single portfolio
    simulation without using the CLI. It handles all setup including backend
    resolution, RNG initialization, and parameter building.

    Args:
        config: Model configuration specifying simulation parameters.
        index_series: Benchmark index returns as a pandas Series.
        options: Optional RunOptions with seed, backend, and config overrides.

    Returns:
        RunArtifacts containing simulation results, summary statistics,
        and metadata for export or further analysis.

    Raises:
        ValueError: If index_series cannot be converted to a pandas Series.

    Example::

        from pa_core.config import load_config
        from pa_core.facade import run_single, RunOptions

        cfg = load_config("config.yml")
        idx = pd.read_csv("index.csv")["returns"]

        # Basic run
        artifacts = run_single(cfg, idx)

        # With seed for reproducibility
        artifacts = run_single(cfg, idx, RunOptions(seed=42))

        # With config overrides
        artifacts = run_single(cfg, idx, RunOptions(
            seed=42,
            config_overrides={"N_SIMULATIONS": 1000}
        ))
    """

    import pandas as pd

    from .agents.registry import build_from_config
    from .backend import resolve_and_set_backend
    from .random import spawn_agent_rngs_with_ids, spawn_rngs
    from .sim import draw_financing_series, draw_joint_returns
    from .sim.covariance import build_cov_matrix
    from .sim.metrics import summary_table
    from .sim.params import (
        build_covariance_return_overrides,
        build_params,
        resolve_covariance_inputs,
    )
    from .sim.regimes import (
        apply_regime_labels,
        build_regime_draw_params,
        resolve_regime_start,
        simulate_regime_paths,
    )
    from .simulations import simulate_agents
    from .units import normalize_return_inputs
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

    return_inputs = normalize_return_inputs(run_cfg)
    sigma_h = float(return_inputs["sigma_H"])
    sigma_e = float(return_inputs["sigma_E"])
    sigma_m = float(return_inputs["sigma_M"])

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
    sigma_vec, corr_mat = resolve_covariance_inputs(
        cov,
        idx_sigma=idx_sigma,
        sigma_h=sigma_h,
        sigma_e=sigma_e,
        sigma_m=sigma_m,
        rho_idx_H=run_cfg.rho_idx_H,
        rho_idx_E=run_cfg.rho_idx_E,
        rho_idx_M=run_cfg.rho_idx_M,
        rho_H_E=run_cfg.rho_H_E,
        rho_H_M=run_cfg.rho_H_M,
        rho_E_M=run_cfg.rho_E_M,
    )

    params = build_params(
        run_cfg,
        mu_idx=mu_idx,
        idx_sigma=float(sigma_vec[0]),
        return_overrides=build_covariance_return_overrides(sigma_vec, corr_mat),
    )

    rng_returns, rng_regime = spawn_rngs(run_options.seed, 2)
    regime_params = None
    regime_paths = None
    regime_labels = None
    if run_cfg.regimes is not None:
        if run_cfg.regime_transition is None:
            raise ValueError("regime_transition is required when regimes are specified")
        regime_params, labels = build_regime_draw_params(
            run_cfg,
            mu_idx=mu_idx,
            idx_sigma=idx_sigma,
            n_samples=n_samples,
        )
        regime_paths = simulate_regime_paths(
            n_sim=run_cfg.N_SIMULATIONS,
            n_months=run_cfg.N_MONTHS,
            transition=run_cfg.regime_transition,
            start_state=resolve_regime_start(run_cfg),
            rng=rng_regime,
        )
        regime_labels = apply_regime_labels(regime_paths, labels)
    r_beta, r_H, r_E, r_M = draw_joint_returns(
        n_months=run_cfg.N_MONTHS,
        n_sim=run_cfg.N_SIMULATIONS,
        params=params,
        rng=rng_returns,
        regime_paths=regime_paths,
        regime_params=regime_params,
    )
    corr_repair_info = params.get("_correlation_repair_info")
    fin_rngs, substream_ids = spawn_agent_rngs_with_ids(
        run_options.seed,
        ["internal", "external_pa", "active_ext"],
        legacy_order=run_options.legacy_agent_rng,
    )
    f_int, f_ext, f_act = draw_financing_series(
        n_months=run_cfg.N_MONTHS,
        n_sim=run_cfg.N_SIMULATIONS,
        params=params,
        financing_mode=run_cfg.financing_mode,
        rngs=fin_rngs,
    )

    agents = build_from_config(run_cfg)
    returns = simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act)
    summary = summary_table(returns, benchmark="Base")
    raw_returns = {name: pd.DataFrame(data) for name, data in returns.items()}
    if regime_labels is not None:
        raw_returns["Regime"] = pd.DataFrame(regime_labels)
    inputs = run_cfg.model_dump()
    if isinstance(corr_repair_info, dict) and corr_repair_info.get("repair_applied"):
        inputs["correlation_repair_applied"] = True
        inputs["correlation_repair_details"] = json.dumps(corr_repair_info)

    manifest = {
        "seed": run_options.seed,
        "substream_ids": substream_ids,
    }

    return RunArtifacts(
        config=run_cfg,
        index_series=idx_series,
        returns=returns,
        summary=summary,
        inputs=inputs,
        raw_returns=raw_returns,
        manifest=manifest,
    )


def run_sweep(
    config: "ModelConfig",
    index_series: "pd.Series",
    sweep_params: Mapping[str, Any] | None = None,
    options: RunOptions | None = None,
) -> SweepArtifacts:
    """Run a parameter sweep with optional overrides and return artifacts.

    Executes simulations across a grid of parameter combinations defined by
    the analysis_mode in the configuration. Supports capital allocation sweeps,
    return parameter sweeps, alpha share sweeps, and volatility multiplier sweeps.

    Args:
        config: Model configuration specifying base simulation parameters.
        index_series: Benchmark index returns as a pandas Series.
        sweep_params: Optional dictionary of sweep-specific parameter overrides
            (e.g., analysis_mode, sweep ranges, step sizes).
        options: Optional RunOptions with seed, backend, and config overrides.

    Returns:
        SweepArtifacts containing results for each parameter combination,
        consolidated summary DataFrame, and metadata.

    Raises:
        ValueError: If index_series cannot be converted to a pandas Series.

    Example::

        from pa_core.config import load_config
        from pa_core.facade import run_sweep, RunOptions

        cfg = load_config("config.yml")
        idx = pd.read_csv("index.csv")["returns"]

        # Run volatility multiplier sweep
        sweep_params = {
            "analysis_mode": "vol_mult",
            "sd_multiple_min": 0.5,
            "sd_multiple_max": 2.0,
            "sd_multiple_step": 0.25,
        }
        artifacts = run_sweep(cfg, idx, sweep_params, RunOptions(seed=42))

        # Access consolidated results
        print(artifacts.summary)
    """

    import pandas as pd

    from .backend import resolve_and_set_backend
    from .random import spawn_agent_rngs_with_ids, spawn_rngs
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
    fin_rngs, substream_ids = spawn_agent_rngs_with_ids(
        run_options.seed,
        ["internal", "external_pa", "active_ext"],
        legacy_order=run_options.legacy_agent_rng,
    )
    results = run_parameter_sweep(
        run_cfg,
        idx_series,
        rng_returns,
        fin_rngs,
        seed=run_options.seed,
    )
    summary = sweep_results_to_dataframe(results)
    inputs = run_cfg.model_dump()

    manifest = {
        "seed": run_options.seed,
        "substream_ids": substream_ids,
    }

    return SweepArtifacts(
        config=run_cfg,
        index_series=idx_series,
        results=results,
        summary=summary,
        inputs=inputs,
        manifest=manifest,
    )


def export(
    artifacts: Union[RunArtifacts, SweepArtifacts],
    output_path: str | Path,
    options: ExportOptions | None = None,
) -> Path:
    """Export simulation artifacts to an output file.

    Writes simulation results to Excel format with configurable options
    for chart embedding, pivoting, and sensitivity analysis inclusion.

    Args:
        artifacts: RunArtifacts or SweepArtifacts from a simulation run.
        output_path: Destination file path (typically .xlsx extension).
        options: Optional ExportOptions controlling export behavior.

    Returns:
        Path to the created output file.

    Raises:
        ValueError: If artifacts type is not recognized.
        OSError: If output file cannot be written.

    Example::

        from pa_core.facade import run_single, export, ExportOptions

        artifacts = run_single(config, index_series)

        # Basic export
        export(artifacts, "results.xlsx")

        # Export with options
        export(artifacts, "results.xlsx", ExportOptions(
            pivot=True,
            include_charts=True,
            alt_text="Portfolio simulation results"
        ))
    """
    from pathlib import Path as PathLib

    from .reporting import export_to_excel
    from .reporting.sweep_excel import export_sweep_results

    export_opts = options or ExportOptions()
    output = PathLib(output_path)

    if isinstance(artifacts, RunArtifacts):
        # Prepare inputs dict with optional internal DataFrames
        inputs_dict: dict[str, Any] = dict(artifacts.inputs)
        metadata = None
        if artifacts.manifest is not None:
            metadata = {
                "rng_seed": artifacts.manifest.get("seed"),
                "substream_ids": artifacts.manifest.get("substream_ids"),
            }

        export_to_excel(
            inputs_dict,
            artifacts.summary,
            artifacts.raw_returns,
            filename=str(output),
            pivot=export_opts.pivot,
            metadata=metadata,
            finalize=True,
        )
    elif isinstance(artifacts, SweepArtifacts):
        metadata = None
        if artifacts.manifest is not None:
            metadata = {
                "rng_seed": artifacts.manifest.get("seed"),
                "substream_ids": artifacts.manifest.get("substream_ids"),
            }
        export_sweep_results(artifacts.results, filename=str(output), metadata=metadata)
    else:
        raise ValueError(f"Unsupported artifacts type: {type(artifacts)}")

    return output
