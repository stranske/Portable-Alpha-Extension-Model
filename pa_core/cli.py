"""Command-line interface for running simulations.

Additional options allow exporting visualisations and launching the
Streamlit dashboard after a run.

CLI flags:
    --png / --pdf / --pptx  Static exports (can be combined)
    --html                 Save interactive HTML
    --gif                  Animated export of monthly paths
    --alt-text TEXT        Alt text for HTML/PPTX exports
    --packet               Committee-ready export packet (PPTX + Excel)
    --dashboard            Launch Streamlit dashboard after run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

# Fix UTF-8 encoding for Windows compatibility
if sys.platform.startswith("win"):
    import os

    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# Rich is imported lazily in functions to keep import time low

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# Intentionally avoid heavy imports at module import time. Required modules are
# imported lazily inside functions after environment bootstrap.

# Placeholders for late-bound globals assigned in main()
draw_joint_returns: Any = None
draw_financing_series: Any = None
simulate_agents: Any = None
export_to_excel: Any = None
build_from_config: Any = None
build_cov_matrix: Any = None
create_export_packet: Any = None

# Configure logger for this module
logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    """Format logs as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        entry = {
            "level": record.levelname,
            "timestamp": ts,
            "module": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(entry)


def create_enhanced_summary(
    returns_map: dict[str, "np.ndarray"],
    *,
    benchmark: str | None = None,
) -> "pd.DataFrame":
    """Create summary table with standard breach and shortfall defaults."""

    # Local import to avoid heavy imports at module load
    from .sim.metrics import summary_table

    return summary_table(returns_map, benchmark=benchmark)


def print_enhanced_summary(summary: "pd.DataFrame") -> None:
    """Print enhanced summary with explanations."""
    # Local imports to avoid heavy import at module load
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from .reporting.console import print_summary

    console = Console()

    # Print explanatory header
    explanation = Text()
    explanation.append("Portfolio Analysis Results\n", style="bold blue")
    explanation.append("Metrics Explanation:\n", style="bold")
    explanation.append("• AnnReturn: Annualized return (%)\n")
    explanation.append("• AnnVol: Annualized volatility (%)\n")
    explanation.append("• VaR: Value at Risk (95% confidence)\n")
    explanation.append("• BreachProb: Probability of monthly loss > 2%\n")
    if "ShortfallProb" in summary.columns:
        explanation.append("• ShortfallProb: Probability of annual loss > 5%\n")
    explanation.append("• TE: Tracking Error vs benchmark\n")

    console.print(Panel(explanation, title="Understanding Your Results"))

    # Print the table
    print_summary(summary)

    # Print additional guidance
    guidance = Text()
    guidance.append("\n💡 Interpretation Tips:\n", style="bold green")
    guidance.append("• Lower ShortfallProb is better (< 5% is typically good)\n")
    guidance.append(
        "• Higher AnnReturn with lower AnnVol indicates better risk-adjusted returns\n"
    )
    guidance.append("• TE shows how much each strategy deviates from the benchmark\n")

    console.print(guidance)


class Dependencies:
    """Container for CLI dependencies using explicit dependency injection."""

    def __init__(
        self,
        build_from_config=None,
        export_to_excel=None,
        draw_financing_series=None,
        draw_joint_returns=None,
        build_cov_matrix=None,
        simulate_agents=None,
    ):
        """Initialize dependencies with explicit function parameters.

        Args:
            build_from_config: Function to build agents from config
            export_to_excel: Function to export results to Excel
            draw_financing_series: Function to generate financing series
            draw_joint_returns: Function to generate joint returns
            build_cov_matrix: Function to build covariance matrix
            simulate_agents: Function to simulate agents

        If any parameter is None, the default implementation will be imported.
        """
        # Import defaults only when needed to avoid heavy imports at module load
        if build_from_config is None:
            from .agents.registry import build_from_config
        if export_to_excel is None:
            from .reporting import export_to_excel
        if draw_financing_series is None:
            from .sim import draw_financing_series
        if draw_joint_returns is None:
            from .sim import draw_joint_returns
        if build_cov_matrix is None:
            from .sim.covariance import build_cov_matrix
        if simulate_agents is None:
            from .simulations import simulate_agents

        self.build_from_config = build_from_config
        self.export_to_excel = export_to_excel
        self.draw_financing_series = draw_financing_series
        self.draw_joint_returns = draw_joint_returns
        self.build_cov_matrix = build_cov_matrix
        self.simulate_agents = simulate_agents


def main(
    argv: Optional[Sequence[str]] = None, deps: Optional[Dependencies] = None
) -> None:
    # Lightweight bootstrap: ensure numpy is available; if not, try to re-exec using
    # the project's virtualenv interpreter to satisfy subprocess tests that use `python`.
    import os
    import sys

    try:  # quick probe for required heavy deps in subprocess execution
        import numpy as _np  # noqa: F401
        import pandas as _pd  # noqa: F401
    except Exception:  # pragma: no cover - only triggered in misconfigured subprocs
        # Attempt to locate project venv based on package location
        project_root = Path(__file__).resolve().parents[1]
        venv_python = project_root / ".venv" / "bin" / "python"
        if venv_python.exists() and str(venv_python) != sys.executable:
            # Re-exec under the venv interpreter, preserving args
            args_list = list(argv) if argv is not None else sys.argv[1:]
            os.execv(
                str(venv_python), [str(venv_python), "-m", "pa_core.cli", *args_list]
            )
        # If no venv found, continue and let normal imports raise a helpful error later

    # Import light dependencies needed for argument parsing defaults
    # Import pandas for runtime usage (safe after bootstrap probe above)
    import pandas as pd  # type: ignore

    from .stress import STRESS_PRESETS

    parser = argparse.ArgumentParser(description="Portable Alpha simulation")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--index", required=True, help="Index returns CSV")
    parser.add_argument("--output", default="Outputs.xlsx", help="Output workbook")
    parser.add_argument(
        "--mode",
        choices=["capital", "returns", "alpha_shares", "vol_mult"],
        default="returns",
        help="Parameter sweep analysis mode",
    )
    parser.add_argument(
        "--stress-preset",
        choices=sorted(STRESS_PRESETS.keys()),
        help="Apply predefined stress scenario",
    )
    parser.add_argument(
        "--pivot",
        action="store_true",
        help="Write all raw returns in a single long-format sheet",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "cupy"],
        help="Computation backend",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Write structured JSON logs to runs/<timestamp>/run.log and reference from manifest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulations",
    )
    parser.add_argument("--png", action="store_true", help="Export PNG chart")
    parser.add_argument("--pdf", action="store_true", help="Export PDF chart")
    parser.add_argument(
        "--pptx",
        action="store_true",
        help="Export PPTX file with charts",
    )
    parser.add_argument("--html", action="store_true", help="Export HTML chart")
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Export GIF animation of monthly paths",
    )
    parser.add_argument(
        "--alt-text",
        dest="alt_text",
        help="Alt text for HTML/PPTX exports",
    )
    parser.add_argument(
        "--packet",
        action="store_true",
        help="Export comprehensive committee packet (PPTX + Excel)",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run one-factor sensitivity analysis on key parameters and include a tornado chart in packet/Excel exports",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard after run",
    )
    parser.add_argument(
        "--prev-manifest",
        dest="prev_manifest",
        help="Path to manifest.json from previous run for diff",
    )
    parser.add_argument(
        "--suggest-sleeves",
        action="store_true",
        help="Suggest feasible sleeve allocations before running",
    )
    parser.add_argument(
        "--tradeoff-table",
        action="store_true",
        help="Compute sleeve trade-off table and include in Excel/packet",
    )
    parser.add_argument(
        "--tradeoff-top",
        type=int,
        default=10,
        help="Top-N rows to include in the trade-off table",
    )
    parser.add_argument(
        "--tradeoff-sort",
        type=str,
        default="risk_score",
        help="Column to sort trade-off table by (e.g., risk_score, ExternalPA_TE)",
    )
    parser.add_argument(
        "--max-te",
        type=float,
        default=0.02,
        help="Maximum tracking error for sleeve suggestions",
    )
    parser.add_argument(
        "--max-breach",
        type=float,
        default=0.05,
        help="Maximum breach probability for sleeve suggestions",
    )
    parser.add_argument(
        "--max-cvar",
        type=float,
        default=0.03,
        help="Maximum CVaR for sleeve suggestions",
    )
    parser.add_argument(
        "--sleeve-step",
        type=float,
        default=0.25,
        help="Grid step size for sleeve suggestions",
    )
    # Optional sleeve bounds (in capital mm units)
    parser.add_argument(
        "--min-external",
        type=float,
        default=None,
        help="Minimum ExternalPA capital (mm)",
    )
    parser.add_argument(
        "--max-external",
        type=float,
        default=None,
        help="Maximum ExternalPA capital (mm)",
    )
    parser.add_argument(
        "--min-active", type=float, default=None, help="Minimum ActiveExt capital (mm)"
    )
    parser.add_argument(
        "--max-active", type=float, default=None, help="Maximum ActiveExt capital (mm)"
    )
    parser.add_argument(
        "--min-internal",
        type=float,
        default=None,
        help="Minimum InternalPA capital (mm)",
    )
    parser.add_argument(
        "--max-internal",
        type=float,
        default=None,
        help="Maximum InternalPA capital (mm)",
    )
    args = parser.parse_args(argv)

    prev_manifest_data: dict[str, Any] | None = None
    prev_summary_df: pd.DataFrame = pd.DataFrame()
    if getattr(args, "prev_manifest", None):
        try:
            prev_manifest_path = Path(args.prev_manifest)
            if prev_manifest_path.exists():
                prev_manifest_data = json.loads(prev_manifest_path.read_text())
                prev_out = (
                    prev_manifest_data.get("cli_args", {}).get("output")
                    if isinstance(prev_manifest_data, dict)
                    else None
                )
                if prev_out and Path(prev_out).exists():
                    try:
                        prev_summary_df = pd.read_excel(prev_out, sheet_name="Summary")
                    except Exception:
                        prev_summary_df = pd.DataFrame()
        except Exception:
            prev_manifest_data = None
            prev_summary_df = pd.DataFrame()

    # Defer heavy imports until after bootstrap (lightweight imports only)
    from .backend import resolve_and_set_backend
    from .config import load_config

    cfg = load_config(args.config)
    # Resolve and set backend once, with proper signature
    backend_choice = resolve_and_set_backend(args.backend, cfg)
    args.backend = backend_choice

    # Echo backend selection at start
    print(f"[BACKEND] Using backend: {backend_choice}")

    from .data import load_index_returns
    from .logging_utils import setup_json_logging
    from .manifest import ManifestWriter
    from .random import spawn_agent_rngs, spawn_rngs
    from .reporting.attribution import (
        compute_sleeve_return_attribution,
        compute_sleeve_risk_attribution,
    )
    from .reporting.sweep_excel import export_sweep_results
    from .run_flags import RunFlags
    from .sleeve_suggestor import suggest_sleeve_sizes
    from .stress import apply_stress_preset
    from .sweep import run_parameter_sweep
    from .viz.utils import safe_to_numpy

    # Initialize dependencies - use provided deps for testing or create default
    if deps is None:
        deps = Dependencies(
            build_from_config=build_from_config,
            export_to_excel=export_to_excel,
            draw_financing_series=draw_financing_series,
            draw_joint_returns=draw_joint_returns,
            build_cov_matrix=build_cov_matrix,
            simulate_agents=simulate_agents,
        )

    flags = RunFlags(
        save_xlsx=args.output,
        png=args.png,
        pdf=args.pdf,
        pptx=args.pptx,
        html=args.html,
        gif=args.gif,
        dashboard=args.dashboard,
        alt_text=args.alt_text,
        packet=args.packet,
    )

    # cfg is already loaded earlier; backend already resolved

    # Optional structured logging setup
    run_log_path: Path | None = None
    if args.log_json:
        # Create run directory under ./runs/<timestamp>
        ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path("runs") / ts
        run_log_path = run_dir / "run.log"
        try:
            setup_json_logging(str(run_log_path))
        except (OSError, PermissionError, RuntimeError, ValueError) as e:
            logger.warning(f"Failed to set up JSON logging: {e}")

    rng_returns = spawn_rngs(args.seed, 1)[0]
    fin_rngs = spawn_agent_rngs(
        args.seed,
        ["internal", "external_pa", "active_ext"],
    )

    if args.mode is not None:
        cfg = cfg.model_copy(update={"analysis_mode": args.mode})
    if args.stress_preset:
        cfg = apply_stress_preset(cfg, args.stress_preset)

    # Capture raw params BEFORE any config modifications
    raw_params = cfg.model_dump()

    idx_series = load_index_returns(args.index)

    # Ensure idx_series is a pandas Series for type safety
    if isinstance(idx_series, pd.DataFrame):
        idx_series = idx_series.squeeze()
        if not isinstance(idx_series, pd.Series):
            raise ValueError("Index data must be convertible to pandas Series")
    elif not isinstance(idx_series, pd.Series):
        raise ValueError("Index data must be a pandas Series")

    # Handle sleeve suggestion if requested
    if args.suggest_sleeves:
        suggestions = suggest_sleeve_sizes(
            cfg,
            idx_series,
            max_te=args.max_te,
            max_breach=args.max_breach,
            max_cvar=args.max_cvar,
            step=args.sleeve_step,
            min_external=args.min_external,
            max_external=args.max_external,
            min_active=args.min_active,
            max_active=args.max_active,
            min_internal=args.min_internal,
            max_internal=args.max_internal,
            seed=args.seed,
        )
        if suggestions.empty:
            print("No feasible sleeve allocations found.")
            return
        print(suggestions.to_string(index=True))
        choice = input(
            "Select row index to apply and continue (blank to abort): "
        ).strip()
        if not choice:
            print("Aborting run.")
            return
        try:
            idx_sel = int(choice)
            row = suggestions.iloc[idx_sel]
        except (ValueError, IndexError):
            print("Invalid selection. Aborting run.")
            return
        cfg = cfg.model_copy(
            update={
                # Direct float conversion for clarity and efficiency
                "external_pa_capital": float(row["external_pa_capital"]),
                "active_ext_capital": float(row["active_ext_capital"]),
                "internal_pa_capital": float(row["internal_pa_capital"]),
            }
        )

    if (
        cfg.analysis_mode in ["capital", "returns", "alpha_shares", "vol_mult"]
        and not args.sensitivity
    ):
        # Parameter sweep mode
        results = run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)
        export_sweep_results(results, filename=args.output)

        # Write reproducibility manifest
        mw = ManifestWriter(Path(args.output).with_name("manifest.json"))
        # Only include args.output in data_files if it exists
        data_files = [args.index, args.config]
        if args.output and Path(args.output).exists():
            data_files.append(args.output)
        mw.write(
            config_path=args.config,
            data_files=data_files,
            seed=args.seed,
            cli_args=vars(args),
            backend=args.backend,
            run_log=str(run_log_path) if run_log_path else None,
            previous_run=args.prev_manifest,
        )
        manifest_json = Path(args.output).with_name("manifest.json")
        manifest_data = None
        try:
            if manifest_json.exists():
                manifest_data = json.loads(manifest_json.read_text())
        except (json.JSONDecodeError, FileNotFoundError, PermissionError):
            manifest_data = None

        # Handle packet export for parameter sweep mode
        if flags.packet:
            try:
                from . import viz

                # Build consolidated summary from sweep results (similar to export_sweep_results)
                summary_frames = []
                for res in results:
                    summary = res["summary"].copy()
                    summary["ShortfallProb"] = summary.get("ShortfallProb", 0.0)
                    summary["Combination"] = f"Run{res['combination_id']}"
                    summary_frames.append(summary)

                if summary_frames:
                    from .reporting.export_packet import (
                        create_export_packet as create_export_packet_fn,
                    )

                    all_summary = pd.concat(summary_frames, ignore_index=True)

                    # Create visualization from consolidated summary
                    if "ShortfallProb" in all_summary.columns:
                        fig = viz.risk_return.make(all_summary)
                    else:
                        fig = viz.sharpe_ladder.make(all_summary)

                    # Create export packet with sweep results
                    base_name = Path(args.output or "parameter_sweep_packet").stem

                    # Create a simplified raw_returns_dict for packet export
                    raw_returns_dict = {"Summary": all_summary}

                    pptx_path, excel_path = create_export_packet_fn(
                        figs=[fig],
                        summary_df=all_summary,
                        raw_returns_dict=raw_returns_dict,
                        inputs_dict={k: raw_params.get(k, "") for k in raw_params},
                        base_filename=base_name,
                        alt_texts=[flags.alt_text] if flags.alt_text else None,
                        pivot=args.pivot,
                        manifest=manifest_data,
                    )
                    print("✅ Parameter sweep export packet created:")
                    print(f"   📊 Excel: {excel_path}")
                    print(f"   📋 PowerPoint: {pptx_path}")
                else:
                    print("⚠️  No summary data available for export packet")
            except RuntimeError as e:
                print(f"❌ Export packet failed: {e}")
            except (ImportError, ModuleNotFoundError) as e:
                logger.error(f"Export packet failed due to missing dependency: {e}")
                print(f"❌ Export packet failed due to missing dependency: {e}")
                print(
                    "💡 Install required packages: pip install plotly kaleido openpyxl"
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Export packet failed due to data issue: {e}")
                print(f"❌ Export packet failed due to data issue: {e}")
                print("💡 Check your configuration and data inputs")

        # Sensitivity analysis can also be applied to parameter sweep results
        if args.sensitivity:
            print("\n🔍 Parameter sweep sensitivity analysis:")
            print("ℹ️  Sensitivity analysis on parameter sweep results shows")
            print("   how different parameter combinations affect outcomes.")

            if results:
                sweep_df = pd.concat(
                    [res["summary"] for res in results], ignore_index=True
                )
                base_agents = sweep_df[sweep_df["Agent"] == "Base"]
                if not base_agents.empty and isinstance(base_agents, pd.DataFrame):
                    best_combo = base_agents.loc[base_agents["AnnReturn"].idxmax()]
                    worst_combo = base_agents.loc[base_agents["AnnReturn"].idxmin()]
                    print(
                        f"   📈 Best combination: {best_combo['AnnReturn']:.2f}% AnnReturn"
                    )
                    print(
                        f"   📉 Worst combination: {worst_combo['AnnReturn']:.2f}% AnnReturn"
                    )
                    print(
                        f"   📊 Range: {best_combo['AnnReturn'] - worst_combo['AnnReturn']:.2f}% difference"
                    )
                else:
                    print("   ⚠️  No Base agent results found in sweep")
            else:
                print("   ❌ No sweep results available")

        return

    # Normal single-run mode below
    mu_idx = float(idx_series.mean())
    idx_sigma = float(idx_series.std(ddof=1))
    mu_H = cfg.mu_H
    sigma_H = cfg.sigma_H
    mu_E = cfg.mu_E
    sigma_E = cfg.sigma_E
    mu_M = cfg.mu_M
    sigma_M = cfg.sigma_M

    # Build covariance (validates shapes)
    _ = deps.build_cov_matrix(
        cfg.rho_idx_H,
        cfg.rho_idx_E,
        cfg.rho_idx_M,
        cfg.rho_H_E,
        cfg.rho_H_M,
        cfg.rho_E_M,
        idx_sigma,
        sigma_H,
        sigma_E,
        sigma_M,
    )

    params = {
        "mu_idx_month": mu_idx / 12,
        "default_mu_H": mu_H / 12,
        "default_mu_E": mu_E / 12,
        "default_mu_M": mu_M / 12,
        "idx_sigma_month": idx_sigma / 12,
        "default_sigma_H": sigma_H / 12,
        "default_sigma_E": sigma_E / 12,
        "default_sigma_M": sigma_M / 12,
        "rho_idx_H": cfg.rho_idx_H,
        "rho_idx_E": cfg.rho_idx_E,
        "rho_idx_M": cfg.rho_idx_M,
        "rho_H_E": cfg.rho_H_E,
        "rho_H_M": cfg.rho_H_M,
        "rho_E_M": cfg.rho_E_M,
        "internal_financing_mean_month": cfg.internal_financing_mean_month,
        "internal_financing_sigma_month": cfg.internal_financing_sigma_month,
        "internal_spike_prob": cfg.internal_spike_prob,
        "internal_spike_factor": cfg.internal_spike_factor,
        "ext_pa_financing_mean_month": cfg.ext_pa_financing_mean_month,
        "ext_pa_financing_sigma_month": cfg.ext_pa_financing_sigma_month,
        "ext_pa_spike_prob": cfg.ext_pa_spike_prob,
        "ext_pa_spike_factor": cfg.ext_pa_spike_factor,
        "act_ext_financing_mean_month": cfg.act_ext_financing_mean_month,
        "act_ext_financing_sigma_month": cfg.act_ext_financing_sigma_month,
        "act_ext_spike_prob": cfg.act_ext_spike_prob,
        "act_ext_spike_factor": cfg.act_ext_spike_factor,
    }

    N_SIMULATIONS = cfg.N_SIMULATIONS
    N_MONTHS = cfg.N_MONTHS

    r_beta, r_H, r_E, r_M = deps.draw_joint_returns(
        n_months=N_MONTHS,
        n_sim=N_SIMULATIONS,
        params=params,
        rng=rng_returns,
    )
    f_int, f_ext, f_act = deps.draw_financing_series(
        n_months=N_MONTHS,
        n_sim=N_SIMULATIONS,
        params=params,
        rngs=fin_rngs,
    )

    # Build agents and run sim
    agents = deps.build_from_config(cfg)
    returns = deps.simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act)

    # Build summary using wrapper (allows tests to mock this safely)
    summary = create_enhanced_summary(returns, benchmark="Base")
    inputs_dict: dict[str, object] = {k: raw_params.get(k, "") for k in raw_params}
    raw_returns_dict = {k: pd.DataFrame(v) for k, v in returns.items()}

    # Optional attribution tables for downstream exports
    try:
        inputs_dict["_attribution_df"] = compute_sleeve_return_attribution(
            cfg, idx_series
        )
    except (AttributeError, TypeError):  # narrow exceptions required by tests
        # Fallback: aggregate total annualised return by agent if detailed attribution fails
        try:
            rows: list[dict[str, object]] = []
            for agent, arr in returns.items():
                mean_month = float(arr.mean())
                ann = 12.0 * mean_month
                rows.append({"Agent": agent, "Sub": "Total", "Return": ann})
            inputs_dict["_attribution_df"] = pd.DataFrame(rows)
        except (AttributeError, ValueError, TypeError, KeyError) as e2:
            logger.debug(f"Attribution fallback unavailable: {e2}")
            inputs_dict["_attribution_df"] = pd.DataFrame(
                [{"Agent": "", "Sub": "", "Return": 0.0}]
            ).head(0)
    try:
        inputs_dict["_risk_attr_df"] = compute_sleeve_risk_attribution(cfg, idx_series)
    except (AttributeError, ValueError, TypeError, KeyError) as e:
        logger.debug(f"Risk attribution unavailable: {e}")
    print_enhanced_summary(summary)
    # Optional: compute trade-off table (non-interactive) and attach for export
    if args.tradeoff_table:
        try:
            trade_df = suggest_sleeve_sizes(
                cfg,
                idx_series,
                max_te=args.max_te,
                max_breach=args.max_breach,
                max_cvar=args.max_cvar,
                step=args.sleeve_step,
                min_external=args.min_external,
                max_external=args.max_external,
                min_active=args.min_active,
                max_active=args.max_active,
                min_internal=args.min_internal,
                max_internal=args.max_internal,
                seed=args.seed,
                sort_by=args.tradeoff_sort,
            )
            if not trade_df.empty:
                inputs_dict["_tradeoff_df"] = trade_df.head(
                    max(1, args.tradeoff_top)
                ).reset_index(drop=True)
        except Exception as e:
            # Local import to avoid heavy import at module load
            from rich.console import Console
            from rich.panel import Panel

            Console().print(
                Panel(
                    f"[bold yellow]Warning:[/bold yellow] Trade-off table computation failed.\n[dim]Reason: {e}[/dim]",
                    title="Trade-off Table",
                    style="yellow",
                )
            )
    # Optional sensitivity analysis (one-factor deltas on AnnReturn)
    if args.sensitivity:
        try:
            from .sim.sensitivity import one_factor_deltas as sim_one_factor_deltas

            # Build a simple evaluator: change a single param, re-run summary AnnReturn for Base
            base_params = {
                "mu_H": cfg.mu_H,
                "sigma_H": cfg.sigma_H,
                "mu_E": cfg.mu_E,
                "sigma_E": cfg.sigma_E,
                "mu_M": cfg.mu_M,
                "sigma_M": cfg.sigma_M,
                "w_beta_H": cfg.w_beta_H,
                "w_alpha_H": cfg.w_alpha_H,
            }
            steps = {
                "mu_H": 0.01,
                "sigma_H": 0.005,
                "mu_E": 0.01,
                "sigma_E": 0.005,
                "mu_M": 0.01,
                "sigma_M": 0.005,
                "w_beta_H": 0.05,
                "w_alpha_H": 0.05,
            }

            def _eval(p: dict[str, float]) -> float:
                # Copy cfg with updates
                mod_cfg = cfg.model_copy(update=p)
                # Recompute params and draws quickly with same RNGs
                mu_idx_val = inputs_dict.get("mu_idx", 0.06)
                idx_sigma_val = inputs_dict.get("sigma_idx", 0.16)
                try:
                    if isinstance(mu_idx_val, (float, int)):
                        mu_idx = float(mu_idx_val)
                    elif isinstance(mu_idx_val, str):
                        mu_idx = float(mu_idx_val)
                    else:
                        mu_idx = 0.06
                except Exception:
                    mu_idx = 0.06
                try:
                    if isinstance(idx_sigma_val, (float, int)):
                        idx_sigma = float(idx_sigma_val)
                    elif isinstance(idx_sigma_val, str):
                        idx_sigma = float(idx_sigma_val)
                    else:
                        idx_sigma = 0.16
                except Exception:
                    idx_sigma = 0.16
                sigma_H = mod_cfg.sigma_H
                sigma_E = mod_cfg.sigma_E
                sigma_M = mod_cfg.sigma_M
                mu_H = mod_cfg.mu_H
                mu_E = mod_cfg.mu_E
                mu_M = mod_cfg.mu_M
                # Note: We rely on draw_joint_returns to rebuild the covariance from params,
                # so we don't need to materialize the covariance matrix here.
                params_local = {
                    "mu_idx_month": mu_idx / 12,
                    "default_mu_H": mu_H / 12,
                    "default_mu_E": mu_E / 12,
                    "default_mu_M": mu_M / 12,
                    "idx_sigma_month": idx_sigma / 12,
                    "default_sigma_H": sigma_H / 12,
                    "default_sigma_E": sigma_E / 12,
                    "default_sigma_M": sigma_M / 12,
                    "rho_idx_H": mod_cfg.rho_idx_H,
                    "rho_idx_E": mod_cfg.rho_idx_E,
                    "rho_idx_M": mod_cfg.rho_idx_M,
                    "rho_H_E": mod_cfg.rho_H_E,
                    "rho_H_M": mod_cfg.rho_H_M,
                    "rho_E_M": mod_cfg.rho_E_M,
                    # financing left the same for speed
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
                r_beta_l, r_H_l, r_E_l, r_M_l = deps.draw_joint_returns(
                    n_months=mod_cfg.N_MONTHS,
                    n_sim=mod_cfg.N_SIMULATIONS,
                    params=params_local,
                    rng=rng_returns,
                )
                f_int_l, f_ext_l, f_act_l = f_int, f_ext, f_act
                agents_l = deps.build_from_config(mod_cfg)
                returns_l = deps.simulate_agents(
                    agents_l, r_beta_l, r_H_l, r_E_l, r_M_l, f_int_l, f_ext_l, f_act_l
                )
                summary_l = create_enhanced_summary(returns_l, benchmark="Base")
                vals = summary_l.loc[summary_l["Agent"] == "Base", "AnnReturn"]
                return float(vals.to_numpy()[0]) if not vals.empty else 0.0

            sens_df = sim_one_factor_deltas(
                params=base_params, steps=steps, evaluator=_eval
            )
            inputs_dict["_sensitivity_df"] = sens_df
        except ImportError as e:
            logger.warning(f"Sensitivity analysis module not available: {e}")
            # Local import to avoid heavy import at module load
            from rich.console import Console
            from rich.panel import Panel

            Console().print(
                Panel(
                    f"[bold red]Error:[/bold red] Sensitivity analysis module not found.\n[dim]Reason: {e}[/dim]",
                    title="Sensitivity Analysis",
                    style="red",
                )
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Sensitivity analysis configuration error: {e}")
            # Local import to avoid heavy import at module load
            from rich.console import Console
            from rich.panel import Panel

            Console().print(
                Panel(
                    f"[bold yellow]Warning:[/bold yellow] Sensitivity analysis failed due to configuration error.\n[dim]Reason: {e}[/dim]\n[dim]Check parameter names and values in your configuration.[/dim]",
                    title="Sensitivity Analysis",
                    style="yellow",
                )
            )
        except TypeError as e:
            logger.error(f"Sensitivity analysis data type error: {e}")
            # Local import to avoid heavy import at module load
            from rich.console import Console
            from rich.panel import Panel

            Console().print(
                Panel(
                    f"[bold yellow]Warning:[/bold yellow] Sensitivity analysis failed due to data type error.\n[dim]Reason: {e}[/dim]\n[dim]Check that all parameters are numeric values.[/dim]",
                    title="Sensitivity Analysis",
                    style="yellow",
                )
            )

    deps.export_to_excel(
        inputs_dict,
        summary,
        raw_returns_dict,
        filename=flags.save_xlsx or "Outputs.xlsx",
        pivot=args.pivot,
    )

    # Write reproducibility manifest for normal run
    try:
        mw = ManifestWriter(
            Path(flags.save_xlsx or "Outputs.xlsx").with_name("manifest.json")
        )
        data_files = [args.index, args.config]
        out_path = Path(flags.save_xlsx or "Outputs.xlsx")
        if out_path.exists():
            data_files.append(str(out_path))
        mw.write(
            config_path=args.config,
            data_files=data_files,
            seed=args.seed,
            cli_args=vars(args),
            backend=args.backend,
            run_log=str(run_log_path) if run_log_path else None,
            previous_run=args.prev_manifest,
        )
    except (OSError, PermissionError, FileNotFoundError) as e:
        logger.warning(f"Failed to write manifest: {e}")

    # Optional sensitivity analysis (one-factor deltas on AnnReturn)
    if args.sensitivity:
        try:
            from .sensitivity import one_factor_deltas as simple_one_factor_deltas

            print("\n🔍 Running sensitivity analysis...")

            # Build a simple evaluator: change a single param, re-run summary AnnReturn for Base
            def _eval(p: dict[str, float]) -> float:
                """Evaluate AnnReturn for Base agent given parameter overrides."""
                mod_cfg = cfg.model_copy(update=p)

                # Rebuild covariance matrix with new parameters
                deps.build_cov_matrix(
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

                params_local = {
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

                r_beta_l, r_H_l, r_E_l, r_M_l = deps.draw_joint_returns(
                    n_months=mod_cfg.N_MONTHS,
                    n_sim=mod_cfg.N_SIMULATIONS,
                    params=params_local,
                    rng=rng_returns,
                )
                # Reuse existing financing draws for speed in sensitivity.
                # NOTE: This introduces correlation between sensitivity analysis runs,
                # as all runs use the same random financing draws. This is intentional
                # to isolate the effect of parameter changes and reduce noise from
                # random variation. If independent draws are required for each run,
                # modify this section to generate new draws per run. Interpret results
                # accordingly, as sensitivity estimates may be affected by this choice.
                f_int_l, f_ext_l, f_act_l = f_int, f_ext, f_act
                agents_l = deps.build_from_config(mod_cfg)
                returns_l = deps.simulate_agents(
                    agents_l, r_beta_l, r_H_l, r_E_l, r_M_l, f_int_l, f_ext_l, f_act_l
                )
                summary_l = create_enhanced_summary(returns_l, benchmark="Base")
                base_row = summary_l[summary_l["Agent"] == "Base"]
                if isinstance(base_row, pd.DataFrame) and not base_row.empty:
                    return float(base_row["AnnReturn"].iloc[0])
                return 0.0

            # Define parameter perturbations to test (±5% relative changes)
            base_params = {
                "mu_H": cfg.mu_H,
                "sigma_H": cfg.sigma_H,
                "mu_E": cfg.mu_E,
                "sigma_E": cfg.sigma_E,
                "mu_M": cfg.mu_M,
                "sigma_M": cfg.sigma_M,
            }

            scenarios = {}
            failed_params = []
            skipped_params = []

            for param_name, base_value in base_params.items():
                # Test positive perturbation
                pos_key = f"{param_name}_+5%"
                try:
                    pos_value = base_value * 1.05
                    pos_result = _eval({param_name: pos_value})
                    scenarios[pos_key] = pd.DataFrame({"AnnReturn": [pos_result]})
                except (ValueError, ZeroDivisionError) as e:
                    failed_params.append(f"{pos_key}: Configuration error: {str(e)}")
                    skipped_params.append(pos_key)
                    logger.warning(
                        f"Parameter evaluation failed for {pos_key} due to configuration: {e}"
                    )
                    print(f"⚠️  Parameter evaluation failed for {pos_key}: {e}")
                except (KeyError, TypeError) as e:
                    failed_params.append(f"{pos_key}: Data type error: {str(e)}")
                    skipped_params.append(pos_key)
                    logger.error(
                        f"Parameter evaluation failed for {pos_key} due to data issue: {e}"
                    )
                    print(f"⚠️  Parameter evaluation failed for {pos_key}: {e}")

                # Test negative perturbation
                neg_key = f"{param_name}_-5%"
                try:
                    neg_value = base_value * 0.95
                    neg_result = _eval({param_name: neg_value})
                    scenarios[neg_key] = pd.DataFrame({"AnnReturn": [neg_result]})
                except (ValueError, ZeroDivisionError) as e:
                    failed_params.append(f"{neg_key}: Configuration error: {str(e)}")
                    skipped_params.append(neg_key)
                    logger.warning(
                        f"Parameter evaluation failed for {neg_key} due to configuration: {e}"
                    )
                    print(f"⚠️  Parameter evaluation failed for {neg_key}: {e}")
                except (KeyError, TypeError) as e:
                    failed_params.append(f"{neg_key}: Data type error: {str(e)}")
                    skipped_params.append(neg_key)
                    logger.error(
                        f"Parameter evaluation failed for {neg_key} due to data issue: {e}"
                    )
                    print(f"⚠️  Parameter evaluation failed for {neg_key}: {e}")

            if scenarios:
                base_df = summary[summary["Agent"] == "Base"][["AnnReturn"]]
                if not isinstance(base_df, pd.DataFrame):
                    base_df = pd.DataFrame(base_df)
                deltas = simple_one_factor_deltas(base_df, scenarios, value="AnnReturn")

                print("\n📊 Sensitivity Analysis Results:")
                print("=" * 50)
                for param, delta in deltas.items():
                    direction = "📈" if delta > 0 else "📉"
                    print(f"{direction} {param:20} | Delta: {delta:+8.4f}%")

                if skipped_params:
                    print(
                        f"\n⚠️  Warning: {len(skipped_params)} parameter evaluations failed and were skipped:"
                    )
                    for param in skipped_params:
                        print(f"   • {param}")
                    print(
                        "\n💡 Consider reviewing parameter ranges or model constraints."
                    )

                print(
                    f"\n✅ Sensitivity analysis completed. Evaluated {len(scenarios)} scenarios."
                )
            else:
                print(
                    "❌ All parameter evaluations failed. Sensitivity analysis could not be completed."
                )
                print("\n📋 Failed parameter details:")
                for failure in failed_params:
                    print(f"   • {failure}")

        except ImportError:
            logger.error("Sensitivity analysis module not available")
            print("❌ Sensitivity analysis requires the sensitivity module")
        except (ValueError, KeyError) as e:
            logger.error(f"Sensitivity analysis configuration error: {e}")
            print(f"❌ Sensitivity analysis failed due to configuration error: {e}")
            print("💡 Check your parameter names and values")
        except TypeError as e:
            logger.error(f"Sensitivity analysis data type error: {e}")
            print(f"❌ Sensitivity analysis failed due to data type error: {e}")
            print("💡 Ensure all parameters are numeric values")

    if any(
        [
            flags.png,
            flags.pdf,
            flags.pptx,
            flags.html,
            flags.gif,
            flags.dashboard,
            flags.packet,
        ]
    ):
        pass

    if any([flags.png, flags.pdf, flags.pptx, flags.html, flags.gif, flags.dashboard]):
        from . import viz

        plots = Path("plots")
        plots.mkdir(exist_ok=True)
        # Guard summary type for static checkers
        if isinstance(summary, pd.DataFrame) and ("ShortfallProb" in summary.columns):
            fig = viz.risk_return.make(summary)
        else:
            fig = viz.sharpe_ladder.make(summary)
        stem = plots / "summary"

        # Handle packet export first (comprehensive export)
        if flags.packet:
            try:
                from . import viz

                # Use base filename from --output or default
                base_name = Path(flags.save_xlsx or "committee_packet").stem
                # Load manifest (if written) for embedding
                manifest_json = Path(flags.save_xlsx or "Outputs.xlsx").with_name(
                    "manifest.json"
                )
                manifest_data = None
                try:
                    if manifest_json.exists():
                        manifest_data = json.loads(manifest_json.read_text())
                except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                    manifest_data = None

                # Build list of figures for the packet
                figs = [fig]
                # Optional: Sensitivity tornado
                try:
                    # inputs_dict is a plain dict[str, object]; guard types before use
                    sens_val = inputs_dict.get("_sensitivity_df")
                    sens_df_plot: Optional[pd.DataFrame] = (
                        sens_val if isinstance(sens_val, pd.DataFrame) else None
                    )
                    if sens_df_plot is not None and (not sens_df_plot.empty):
                        if {"Parameter", "DeltaAbs"} <= set(sens_df_plot.columns):
                            series = cast(
                                pd.Series,
                                sens_df_plot.set_index("Parameter")["DeltaAbs"].astype(
                                    float
                                ),
                            )
                            figs.append(
                                viz.tornado.make(series, title="Sensitivity Tornado")
                            )
                except Exception:
                    # Non-fatal; continue without tornado figure
                    pass
                # Optional: Return attribution sunburst
                try:
                    attr_val = inputs_dict.get("_attribution_df")
                    attr_df: Optional[pd.DataFrame] = (
                        attr_val if isinstance(attr_val, pd.DataFrame) else None
                    )
                    if attr_df is not None and (not attr_df.empty):
                        if {"Agent", "Sub", "Return"} <= set(attr_df.columns):
                            figs.append(viz.sunburst.make(attr_df))
                except (AttributeError, TypeError) as e:
                    logger.debug(
                        "Skipping sunburst figure due to data issue", exc_info=e
                    )
                # Late-bind create_export_packet to allow test monkeypatching
                from .reporting.export_packet import (
                    create_export_packet as create_export_packet_fn,
                )

                pptx_path, excel_path = create_export_packet_fn(
                    figs=figs,
                    summary_df=summary,
                    raw_returns_dict=raw_returns_dict,
                    inputs_dict=inputs_dict,
                    base_filename=base_name,
                    alt_texts=[flags.alt_text] if flags.alt_text else None,
                    pivot=args.pivot,
                    manifest=manifest_data,
                    prev_summary_df=prev_summary_df,
                    prev_manifest=prev_manifest_data,
                )
                print("✅ Export packet created:")
                print(f"   📊 Excel: {excel_path}")
                print(f"   📋 PowerPoint: {pptx_path}")
            except RuntimeError as e:
                print(f"❌ Export packet failed: {e}")
                return
            except (ImportError, ModuleNotFoundError) as e:
                logger.error(f"Export packet failed due to missing dependency: {e}")
                print(f"❌ Export packet failed due to missing dependency: {e}")
                print(
                    "💡 Install required packages: pip install plotly kaleido openpyxl python-pptx"
                )
                return
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Export packet failed due to data/config issue: {e}")
                print(
                    f"❌ Export packet failed due to data or configuration issue: {e}"
                )
                print("💡 Check your data inputs and configuration settings")
                return
            except (OSError, PermissionError) as e:
                logger.error(f"Export packet failed due to file system issue: {e}")
                print(f"❌ Export packet failed due to file system issue: {e}")
                print("💡 Check file permissions and available disk space")
                return

        # Individual export formats (with improved error handling)
        if flags.png:
            try:
                fig.write_image(stem.with_suffix(".png"), engine="kaleido")
            except (ImportError, ModuleNotFoundError) as e:
                if "kaleido" in str(e).lower() or "chrome" in str(e).lower():
                    logger.error(f"PNG export failed due to missing dependency: {e}")
                    print("❌ PNG export failed: Kaleido or Chrome/Chromium required")
                    print(
                        "💡 Install with: pip install kaleido (preferred) or sudo apt-get install chromium-browser"
                    )
                else:
                    logger.error(f"PNG export failed due to missing module: {e}")
                    print(f"❌ PNG export failed due to missing dependency: {e}")
            except (OSError, PermissionError) as e:
                logger.error(f"PNG export failed due to file system issue: {e}")
                print(f"❌ PNG export failed: Cannot write file - {e}")
                print("💡 Check file permissions and available disk space")
            except (ValueError, TypeError) as e:
                logger.error(f"PNG export failed due to data issue: {e}")
                print(f"❌ PNG export failed: Invalid data - {e}")
                print("💡 Check your visualization data and parameters")
            except Exception as e:
                logger.error(f"PNG export failed: {e}")
                msg = str(e).lower()
                if any(
                    term in msg
                    for term in ("kaleido", "chrome", "chromium", "cancelled")
                ):
                    print("❌ PNG export failed: Kaleido or Chrome/Chromium required")
                    print(
                        "💡 Install with: pip install kaleido (preferred) or sudo apt-get install chromium-browser"
                    )
                else:
                    print(f"❌ PNG export failed: {e}")
        if flags.pdf:
            try:
                viz.pdf_export.save(fig, str(stem.with_suffix(".pdf")))
            except (ImportError, ModuleNotFoundError) as e:
                if "kaleido" in str(e).lower() or "chrome" in str(e).lower():
                    logger.error(f"PDF export failed due to missing dependency: {e}")
                    print("❌ PDF export failed: Kaleido or Chrome/Chromium required")
                    print(
                        "💡 Install with: pip install kaleido (preferred) or sudo apt-get install chromium-browser"
                    )
                else:
                    logger.error(f"PDF export failed due to missing module: {e}")
                    print(f"❌ PDF export failed due to missing dependency: {e}")
            except (OSError, PermissionError) as e:
                logger.error(f"PDF export failed due to file system issue: {e}")
                print(f"❌ PDF export failed: Cannot write file - {e}")
                print("💡 Check file permissions and available disk space")
            except (ValueError, TypeError) as e:
                logger.error(f"PDF export failed due to data issue: {e}")
                print(f"❌ PDF export failed: Invalid data - {e}")
                print("💡 Check your visualization data and parameters")
        if flags.pptx:
            try:
                viz.pptx_export.save(
                    [fig],
                    str(stem.with_suffix(".pptx")),
                    alt_texts=[flags.alt_text] if flags.alt_text else None,
                )
            except (ImportError, ModuleNotFoundError) as e:
                if "kaleido" in str(e).lower() or "chrome" in str(e).lower():
                    logger.error(f"PPTX export failed due to missing dependency: {e}")
                    print("❌ PPTX export failed: Kaleido or Chrome/Chromium required")
                    print(
                        "💡 Install with: pip install kaleido (preferred) or sudo apt-get install chromium-browser"
                    )
                elif "pptx" in str(e).lower() or "python-pptx" in str(e).lower():
                    logger.error(f"PPTX export failed due to missing python-pptx: {e}")
                    print("❌ PPTX export failed: python-pptx required")
                    print("💡 Install with: pip install python-pptx")
                else:
                    logger.error(f"PPTX export failed due to missing module: {e}")
                    print(f"❌ PPTX export failed due to missing dependency: {e}")
            except (OSError, PermissionError) as e:
                logger.error(f"PPTX export failed due to file system issue: {e}")
                print(f"❌ PPTX export failed: Cannot write file - {e}")
                print("💡 Check file permissions and available disk space")
            except (ValueError, TypeError) as e:
                logger.error(f"PPTX export failed due to data issue: {e}")
                print(f"❌ PPTX export failed: Invalid data - {e}")
                print("💡 Check your visualization data and parameters")
        if flags.html:
            viz.html_export.save(
                fig,
                str(stem.with_suffix(".html")),
                alt_text=flags.alt_text,
            )
        if flags.gif:
            try:
                arr = safe_to_numpy(next(iter(raw_returns_dict.values())))
            except (ValueError, TypeError) as e:
                print(f"❌ GIF export failed: Data conversion error - {e}")
                print("💡 Check that return data contains only numeric values")
                return
            anim = viz.animation.make(arr)
            try:
                anim.write_image(str(plots / "paths.gif"))
            except Exception as e:
                if "Chrome" in str(e) or "Kaleido" in str(e) or "Chromium" in str(e):
                    print("❌ GIF export failed: Chrome/Chromium required")
                    print("💡 Install with: sudo apt-get install chromium-browser")
                else:
                    print(f"❌ GIF export failed: {e}")
        if flags.dashboard:
            import os
            import subprocess
            import sys

            # Use the same Python interpreter with -m streamlit to ensure venv
            try:
                dashboard_path = Path("dashboard/app.py")
                if not dashboard_path.exists():
                    raise FileNotFoundError(
                        f"Dashboard file not found: {dashboard_path}"
                    )

                subprocess.run(
                    [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
                    check=True,
                    cwd=os.getcwd(),
                )
            except FileNotFoundError as e:
                print(f"❌ Dashboard launch failed: {e}")
                print(
                    "💡 Ensure the dashboard files are present in the 'dashboard/' directory."
                )
                return
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Dashboard launch failed with exit code {e.returncode}: {e}"
                )
                print(f"❌ Dashboard launch failed with exit code {e.returncode}")
                print("💡 Common solutions:")
                print("   • Install Streamlit: pip install streamlit")
                print("   • Check if 'dashboard/app.py' is valid Python code")
                print("   • Verify your Python environment is properly configured")
                return
            except ImportError as e:
                logger.error(f"Dashboard launch failed due to missing streamlit: {e}")
                print(f"❌ Dashboard launch failed: Streamlit not available - {e}")
                print("💡 Install Streamlit: pip install streamlit")
                return
            except (OSError, PermissionError) as e:
                logger.error(f"Dashboard launch failed due to system issue: {e}")
                print(f"❌ Dashboard launch failed: System/permission error - {e}")
                print("💡 Check file permissions and system resources")
                return


# (Backward compatibility global variable assignment removed)
if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
