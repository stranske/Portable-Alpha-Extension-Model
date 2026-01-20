"""Facade types for CLI and programmatic run entrypoints.

This module provides a clean programmatic API for running simulations,
decoupling business logic from CLI argument parsing. It enables:

- Direct programmatic access to simulations without CLI
- Standardized artifact types for consistent outputs
- Type-safe configuration with clear interfaces
- The canonical pipeline implementation used by all entrypoints

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

CANONICAL_PIPELINE = "pa_core.facade"


def _serialize_agent_semantics_input(inputs: dict[str, Any]) -> None:
    import pandas as pd
    from pandas.api.types import is_list_like

    def _mapping_is_row(mapping: dict[str, Any]) -> bool:
        for value in mapping.values():
            if isinstance(value, dict):
                return False
            if is_list_like(value) and not isinstance(value, (str, bytes)):
                return False
        return True

    agent_semantics_val = inputs.get("_agent_semantics_df")
    if isinstance(agent_semantics_val, pd.DataFrame):
        inputs["_agent_semantics_df"] = agent_semantics_val.to_dict(orient="records")
        return
    if isinstance(agent_semantics_val, pd.Series):
        inputs["_agent_semantics_df"] = pd.DataFrame([agent_semantics_val]).to_dict(
            orient="records"
        )
        return
    if isinstance(agent_semantics_val, tuple):
        if agent_semantics_val and all(isinstance(item, pd.Series) for item in agent_semantics_val):
            inputs["_agent_semantics_df"] = pd.DataFrame(agent_semantics_val).to_dict(
                orient="records"
            )
        else:
            inputs["_agent_semantics_df"] = list(agent_semantics_val)
        return
    if isinstance(agent_semantics_val, list):
        if agent_semantics_val and all(isinstance(item, pd.Series) for item in agent_semantics_val):
            inputs["_agent_semantics_df"] = pd.DataFrame(agent_semantics_val).to_dict(
                orient="records"
            )
        return
    if isinstance(agent_semantics_val, dict):
        if agent_semantics_val and _mapping_is_row(agent_semantics_val):
            inputs["_agent_semantics_df"] = [agent_semantics_val]
            return
        try:
            inputs["_agent_semantics_df"] = pd.DataFrame(agent_semantics_val).to_dict(
                orient="records"
            )
        except Exception:
            return


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
        return_distribution: Override return distribution ("normal" or "student_t").
        return_t_df: Override Student-t degrees of freedom (requires student_t).
        return_copula: Override return copula ("gaussian" or "t").
        covariance_shrinkage: Covariance shrinkage mode override.
        vol_regime: Volatility regime selection override.
        vol_regime_window: Window length for two-state volatility regime.
        analysis_mode: Parameter sweep analysis mode override.
    """

    seed: int | None = None
    backend: str | None = None
    config_overrides: Mapping[str, Any] | None = None
    legacy_agent_rng: bool = False
    return_distribution: str | None = None
    return_t_df: float | None = None
    return_copula: str | None = None
    covariance_shrinkage: str | None = None
    vol_regime: str | None = None
    vol_regime_window: int | None = None
    analysis_mode: str | None = None


def apply_run_options(
    config: "ModelConfig",
    options: RunOptions | None = None,
    sweep_params: Mapping[str, Any] | None = None,
) -> "ModelConfig":
    """Return config updated with RunOptions and optional sweep params."""

    run_options = options or RunOptions()
    overrides: dict[str, Any] = dict(run_options.config_overrides or {})

    if run_options.return_distribution is not None:
        overrides["return_distribution"] = run_options.return_distribution
    if run_options.return_t_df is not None:
        overrides["return_t_df"] = run_options.return_t_df
    if run_options.return_copula is not None:
        overrides["return_copula"] = run_options.return_copula
    if run_options.covariance_shrinkage is not None:
        overrides["covariance_shrinkage"] = run_options.covariance_shrinkage
    if run_options.vol_regime is not None:
        overrides["vol_regime"] = run_options.vol_regime
    if run_options.vol_regime_window is not None:
        overrides["vol_regime_window"] = run_options.vol_regime_window
    if run_options.analysis_mode is not None:
        overrides["analysis_mode"] = run_options.analysis_mode
    if sweep_params:
        overrides.update(sweep_params)

    if not overrides:
        return config

    data = config.model_dump()
    data.update(overrides)
    return config.__class__.model_validate(data)


@dataclass(slots=True)
class ExportOptions:
    """Configuration options for exporting artifacts.

    Attributes:
        pivot: If True, write all returns in a single long-format sheet.
        include_sensitivity: If True, include sensitivity analysis in exports.
        include_charts: If True, embed charts in Excel output.
        alt_text: Alt text for accessibility in exported charts.
        include_visualizations: If True, generate comparison visualizations.
        viz_output_dir: Optional directory for visualization HTML output.
        show_visualizations: If True, display visualizations interactively.
    """

    pivot: bool = False
    include_sensitivity: bool = False
    include_charts: bool = True
    alt_text: str | None = None
    include_visualizations: bool = False
    viz_output_dir: str | Path | None = None
    show_visualizations: bool = False


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
    from .sim.simulation_initialization import initialize_run_rngs
    from .simulations import simulate_agents
    from .units import normalize_return_inputs
    from .validators import select_vol_regime_sigma

    run_options = options or RunOptions()
    run_cfg = apply_run_options(config, run_options)
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

    rng_bundle = initialize_run_rngs(
        run_options.seed,
        legacy_agent_rng=run_options.legacy_agent_rng,
    )
    rng_returns = rng_bundle.rng_returns
    rng_regime = rng_bundle.rng_regime
    fin_rngs = rng_bundle.rngs_financing
    substream_ids = rng_bundle.substream_ids
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
    _serialize_agent_semantics_input(inputs)
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
    from .sim.simulation_initialization import initialize_sweep_rngs
    from .sweep import run_parameter_sweep, sweep_results_to_dataframe

    run_options = options or RunOptions()
    run_cfg = apply_run_options(config, run_options, sweep_params)
    resolve_and_set_backend(run_options.backend, run_cfg)

    idx_series = index_series
    if isinstance(idx_series, pd.DataFrame):
        idx_series = idx_series.squeeze()
        if not isinstance(idx_series, pd.Series):
            raise ValueError("Index data must be convertible to pandas Series")
    elif not isinstance(idx_series, pd.Series):
        raise ValueError("Index data must be a pandas Series")

    rng_bundle = initialize_sweep_rngs(
        run_options.seed,
        legacy_agent_rng=run_options.legacy_agent_rng,
    )
    rng_returns = rng_bundle.rng_returns
    fin_rngs = rng_bundle.rngs_financing
    substream_ids = rng_bundle.substream_ids
    child_seed = rng_bundle.seed
    results = run_parameter_sweep(
        run_cfg,
        idx_series,
        rng_returns,
        fin_rngs,
        seed=child_seed,
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

        # Export with visualization bundle
        export(artifacts, "results.xlsx", ExportOptions(
            include_visualizations=True,
            viz_output_dir="results_viz"
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
        import pandas as pd

        from .reporting.agent_semantics import build_agent_semantics

        agent_semantics_val = inputs_dict.get("_agent_semantics_df")
        has_serialized = isinstance(agent_semantics_val, (list, tuple, dict)) and bool(
            agent_semantics_val
        )
        serialized_agent_semantics = None
        if isinstance(agent_semantics_val, pd.DataFrame) and not agent_semantics_val.empty:
            serialized_agent_semantics = agent_semantics_val.to_dict(orient="records")
        elif has_serialized:
            serialized_agent_semantics = agent_semantics_val

        if serialized_agent_semantics is None:
            agent_semantics_df = build_agent_semantics(artifacts.config)
            serialized_agent_semantics = agent_semantics_df.to_dict(orient="records")

        inputs_dict["_agent_semantics_df"] = serialized_agent_semantics
        artifacts.inputs["_agent_semantics_df"] = serialized_agent_semantics
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

    if (
        export_opts.include_visualizations
        or export_opts.viz_output_dir is not None
        or export_opts.show_visualizations
    ):
        from . import viz
        from .viz import html_export

        viz_dir = (
            PathLib(export_opts.viz_output_dir)
            if export_opts.viz_output_dir is not None
            else output.parent / f"{output.stem}_viz"
        )
        viz_dir.mkdir(parents=True, exist_ok=True)
        scenarios: list[Any]
        if isinstance(artifacts, RunArtifacts):
            scenarios = [
                {
                    "summary": artifacts.summary,
                    "raw_returns": artifacts.raw_returns,
                    "label": "Run",
                }
            ]
            include_returns = True
        else:
            scenarios = list(artifacts.results)
            include_returns = False
        figs = viz.compare_scenarios(scenarios, include_returns=include_returns)
        for name, fig in figs.items():
            alt_text = None
            if export_opts.alt_text:
                alt_text = f"{export_opts.alt_text} ({name.replace('_', ' ')})"
            html_export.save(fig, viz_dir / f"{name}.html", alt_text=alt_text)
            if export_opts.show_visualizations:
                fig.show()

    return output
