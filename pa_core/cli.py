"""Command-line interface for running simulations.

Additional options allow exporting visualisations and launching the
Streamlit dashboard after a run.

CLI flags:
    --png / --pdf / --pptx  Static exports (can be combined)
    --html                 Save interactive HTML
    --gif                  Animated export of monthly paths
    --alt-text TEXT        Alt text for HTML/PPTX exports
    --packet               Committee-ready export packet (PPTX + Excel)
    --bundle PATH          Write run artifact bundle directory
    --dashboard            Launch Streamlit dashboard after run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence, cast

# Fix UTF-8 encoding for Windows compatibility
if sys.platform.startswith("win"):
    import os

    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

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

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        entry = {
            "level": record.levelname,
            "timestamp": ts,
            "module": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(entry)


class RunTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._start_wall = datetime.now(timezone.utc)

    def elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self._start)

    def snapshot(self) -> dict[str, Any]:
        end_wall = datetime.now(timezone.utc)
        return {
            "duration_seconds": self.elapsed(),
            "started_at": self._start_wall.isoformat(),
            "ended_at": end_wall.isoformat(),
        }


def _read_config_snapshot(path: str | Path) -> str:
    try:
        return Path(path).read_text()
    except (FileNotFoundError, OSError, PermissionError):
        logger.debug("Unable to read config snapshot from %s", path)
        return ""


def _hash_index_series(index_series: "pd.Series") -> str:
    import pandas as pd

    hasher = hashlib.sha256()
    hashed = pd.util.hash_pandas_object(index_series, index=True).to_numpy()
    hasher.update(hashed.tobytes())
    return hasher.hexdigest()


def create_enhanced_summary(
    returns_map: dict[str, "np.ndarray"],
    *,
    benchmark: str | None = None,
) -> "pd.DataFrame":
    """Create a summary table from monthly returns with standard thresholds.

    terminal_AnnReturn/monthly_AnnVol/monthly_TE are annualised from monthly
    returns; breach thresholds apply to monthly returns and shortfall thresholds
    use annualised hurdles.
    """

    # Local import to avoid heavy imports at module load
    from .sim.metrics import summary_table

    return summary_table(returns_map, benchmark=benchmark)


def print_enhanced_summary(summary: "pd.DataFrame") -> None:
    """Print summary with unit-aware explanations for annualised metrics."""
    # Local imports to avoid heavy import at module load
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from .reporting.console import print_summary
    from .units import format_unit_label, get_summary_table_unit, get_threshold_unit

    console = Console()
    summary_unit = get_summary_table_unit()
    unit_label = format_unit_label(summary_unit)
    threshold_units = get_threshold_unit()

    # Print explanatory header
    explanation = Text()
    explanation.append("Portfolio Analysis Results\n", style="bold blue")
    explanation.append("Metrics Explanation:\n", style="bold")
    explanation.append(f"• terminal_AnnReturn: {unit_label} return (%)\n")
    explanation.append(f"• monthly_AnnVol: {unit_label} volatility (%)\n")
    explanation.append("• monthly_VaR: Value at Risk (95% confidence)\n")
    explanation.append(
        f"• monthly_BreachProb: Share of simulated months below the "
        f"{threshold_units['breach_threshold']} breach threshold\n"
    )
    if "terminal_ShortfallProb" in summary.columns:
        explanation.append(
            "• terminal_ShortfallProb: Probability terminal compounded return is below the "
            f"{threshold_units['shortfall_threshold']} threshold\n"
        )
    if "monthly_MaxDD" in summary.columns:
        explanation.append("• monthly_MaxDD: Worst peak-to-trough decline of compounded wealth\n")
    if "monthly_TimeUnderWater" in summary.columns:
        explanation.append(
            "• monthly_TimeUnderWater: Fraction of periods with compounded return below zero\n"
        )
    explanation.append(f"• monthly_TE: {unit_label} active return volatility vs benchmark\n")

    console.print(Panel(explanation, title="Understanding Your Results"))

    # Print the table
    print_summary(summary)

    # Print additional guidance
    guidance = Text()
    guidance.append("\n💡 Interpretation Tips:\n", style="bold green")
    guidance.append(
        "• Lower terminal_ShortfallProb means fewer paths breach the terminal return threshold\n"
    )
    guidance.append(
        "• Higher terminal_AnnReturn with lower monthly_AnnVol indicates better risk-adjusted returns\n"
    )
    guidance.append(
        "• monthly_TE shows how volatile active returns are relative to the benchmark\n"
    )

    console.print(guidance)


def _maybe_print_run_diff(
    *,
    current_manifest: Mapping[str, Any] | None,
    prev_manifest: Mapping[str, Any] | None,
    current_summary: "pd.DataFrame",
    prev_summary: "pd.DataFrame" | None,
) -> None:
    if prev_manifest is None and (prev_summary is None or prev_summary.empty):
        return
    import pandas as pd

    from .reporting.console import print_run_diff
    from .reporting.run_diff import build_run_diff

    prev_summary_df = prev_summary if prev_summary is not None else pd.DataFrame()
    try:
        cfg_diff, metric_diff = build_run_diff(
            current_manifest, prev_manifest, current_summary, prev_summary_df
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Run diff unavailable: {exc}")
        return
    print_run_diff(cfg_diff, metric_diff)


class Dependencies:
    """Container for CLI dependencies using explicit dependency injection."""

    def __init__(
        self,
        build_from_config: Callable[..., Any] | None = None,
        export_to_excel: Callable[..., Any] | None = None,
        draw_financing_series: Callable[..., Any] | None = None,
        draw_joint_returns: Callable[..., Any] | None = None,
        build_cov_matrix: Callable[..., Any] | None = None,
        simulate_agents: Callable[..., Any] | None = None,
    ) -> None:
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
            from .agents.registry import build_from_config as build_from_config_impl

            build_from_config = build_from_config_impl
        if export_to_excel is None:
            from .reporting import export_to_excel as export_to_excel_impl

            export_to_excel = export_to_excel_impl
        if draw_financing_series is None:
            from .sim import draw_financing_series as draw_financing_series_impl

            draw_financing_series = draw_financing_series_impl
        if draw_joint_returns is None:
            from .sim import draw_joint_returns as draw_joint_returns_impl

            draw_joint_returns = draw_joint_returns_impl
        if build_cov_matrix is None:
            from .sim.covariance import build_cov_matrix as build_cov_matrix_impl

            build_cov_matrix = build_cov_matrix_impl
        if simulate_agents is None:
            from .simulations import simulate_agents as simulate_agents_impl

            simulate_agents = simulate_agents_impl

        assert build_from_config is not None
        assert export_to_excel is not None
        assert draw_financing_series is not None
        assert draw_joint_returns is not None
        assert build_cov_matrix is not None
        assert simulate_agents is not None

        self.build_from_config: Callable[..., Any] = build_from_config
        self.export_to_excel: Callable[..., Any] = export_to_excel
        self.draw_financing_series: Callable[..., Any] = draw_financing_series
        self.draw_joint_returns: Callable[..., Any] = draw_joint_returns
        self.build_cov_matrix: Callable[..., Any] = build_cov_matrix
        self.simulate_agents: Callable[..., Any] = simulate_agents


def main(
    argv: Optional[Sequence[str]] = None,
    deps: Optional[Dependencies] = None,
    *,
    emit_deprecation_warning: bool = True,
) -> None:
    # Lightweight bootstrap: ensure numpy is available; if not, try to re-exec using
    # the project's virtualenv interpreter to satisfy subprocess tests that use `python`.
    import os
    import sys
    import warnings

    # `pa run` delegates here with emit_deprecation_warning=False; legacy entrypoints warn.
    # Output strings from this entry point are validated in tests/expected_cli_outputs.py
    # and golden tests, so argument parsing/output ordering should stay stable across all
    # invocation paths (pa run, pa_core.cli, and legacy __main__).
    if emit_deprecation_warning:
        warnings.warn(
            "Direct use of pa_core.cli is deprecated; use `pa run` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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
            os.execv(str(venv_python), [str(venv_python), "-m", "pa_core.cli", *args_list])
        # If no venv found, continue and let normal imports raise a helpful error later

    # Import light dependencies needed for argument parsing defaults
    # Import pandas for runtime usage (safe after bootstrap probe above)
    import pandas as pd

    from .stress import STRESS_PRESETS

    # Argument parsing stays centralized here to preserve documented flags/output behavior.
    # This parser is the canonical run surface referenced by docs and expected output fixtures
    # (tests/expected_cli_outputs.py and golden files), so any stdout/stderr expectations remain
    # consistent with pa_core.__main__ and pa run usage. The downstream flow is:
    # argparse Namespace -> RunOptions -> apply_run_options -> resolve backend -> run_single.
    parser = argparse.ArgumentParser(description="Portable Alpha simulation")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "YAML config file (set financing_mode to broadcast for shared paths or "
            "per_path for independent draws)"
        ),
    )
    parser.add_argument("--index", required=True, help="Index returns CSV")
    parser.add_argument(
        "--index-frequency",
        choices=["daily", "weekly", "monthly", "quarterly"],
        default=None,
        help="Explicitly declare index data frequency (skips auto-detection)",
    )
    parser.add_argument(
        "--resample",
        choices=["monthly"],
        default=None,
        help="Resample index data to target frequency (e.g., daily->monthly)",
    )
    parser.add_argument("--output", default="Outputs.xlsx", help="Output workbook")
    parser.add_argument(
        "--bundle",
        default=None,
        help="Write run artifact bundle directory (opt-in)",
    )
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
        choices=["numpy"],
        help="Computation backend (numpy only; cupy/GPU acceleration is not available)",
    )
    parser.add_argument(
        "--cov-shrinkage",
        choices=["none", "ledoit_wolf"],
        default=None,
        help="Covariance shrinkage mode",
    )
    parser.add_argument(
        "--vol-regime",
        choices=["single", "two_state"],
        default=None,
        help="Volatility regime selection",
    )
    parser.add_argument(
        "--vol-regime-window",
        type=int,
        default=None,
        help="Recent window length (months) for two-state regime",
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
    parser.add_argument(
        "--legacy-agent-rng",
        action="store_true",
        help="Use legacy order-dependent agent RNG streams (defaults to stable name-based streams)",
    )
    parser.add_argument(
        "--return-distribution",
        choices=["normal", "student_t"],
        help="Override return distribution (normal or student_t). student_t adds heavier tails and more compute",
    )
    parser.add_argument(
        "--return-t-df",
        type=float,
        help="Override Student-t degrees of freedom (requires student_t; lower df => heavier tails)",
    )
    parser.add_argument(
        "--return-copula",
        choices=["gaussian", "t"],
        help="Override return copula (gaussian or t). t adds tail dependence and extra compute",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the scenario and print its scenario ID",
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
        "--suggest-apply-index",
        type=int,
        default=None,
        help="Auto-apply a suggested sleeve row index without prompting",
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
        help="Column to sort trade-off table by (e.g., risk_score, ExternalPA_monthly_TE)",
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
        help="Maximum monthly_CVaR for sleeve suggestions",
    )
    parser.add_argument(
        "--max-shortfall",
        type=float,
        default=1.0,
        help="Maximum terminal shortfall probability for sleeve suggestions",
    )
    parser.add_argument(
        "--sleeve-step",
        type=float,
        default=0.25,
        help="Grid step size for sleeve suggestions",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Use optimizer for sleeve suggestions (falls back to grid if unavailable)",
    )
    parser.add_argument(
        "--optimize-objective",
        choices=["total_return", "excess_return"],
        default="total_return",
        help="Objective for sleeve optimization",
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
    # argparse returns a Namespace; translate it into RunOptions/mode routing so the CLI
    # delegates to facade run/export helpers without duplicating simulation logic. This keeps
    # test expectations in tests/expected_cli_outputs.py stable when output lines move.
    args = parser.parse_args(argv)

    run_timer = RunTimer()
    run_end_emitted = False
    run_log_path: Path | None = None
    run_backend: str | None = None
    run_id: str | None = None
    artifact_candidates: list[Path] = []
    manifest_path: Path | None = None
    config_snapshot = _read_config_snapshot(args.config)

    def _record_artifact(path: str | Path | None) -> None:
        if not path:
            return
        artifact_candidates.append(Path(path))

    def _current_duration() -> float:
        return run_timer.elapsed()

    def _collect_artifacts() -> list[str]:
        seen: set[str] = set()
        collected: list[str] = []
        for cand in artifact_candidates:
            if cand.exists():
                value = str(cand)
                if value not in seen:
                    seen.add(value)
                    collected.append(value)
        return collected

    def _resolve_path(path: str | Path | None) -> Path | None:
        if path is None:
            return None
        try:
            return Path(path).resolve()
        except OSError:
            return Path(path)

    def _output_key_for_path(path: Path) -> str:
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return path.name

    def _build_outputs_map(paths: Sequence[str]) -> dict[str, str]:
        outputs: dict[str, str] = {}
        collisions: dict[str, int] = {}
        skip_paths = {
            p for p in (_resolve_path(args.config), _resolve_path(manifest_path)) if p is not None
        }
        for value in paths:
            path = Path(value)
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in skip_paths:
                continue
            key = _output_key_for_path(path)
            if key in outputs:
                count = collisions.get(key, 1) + 1
                collisions[key] = count
                key = f"{path.stem}_{count}{path.suffix}"
            outputs[key] = str(path)
        return outputs

    def _maybe_write_bundle(
        *,
        index_hash: str,
        manifest_data: Mapping[str, Any] | None,
    ) -> None:
        if not args.bundle:
            return
        try:
            from .run_artifact_bundle import RunArtifact, RunArtifactBundle

            outputs = _build_outputs_map(_collect_artifacts())
            artifact = RunArtifact(
                config=config_snapshot,
                index_hash=index_hash,
                seed=args.seed,
                manifest=manifest_data,
                outputs=outputs,
            )
            bundle = RunArtifactBundle(artifact)
            bundle.save(args.bundle)
        except (ImportError, ModuleNotFoundError, OSError, PermissionError, ValueError) as exc:
            logger.warning(f"Failed to write artifact bundle: {exc}")

    def _finalize_manifest_timing(
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timing = snapshot or run_timer.snapshot()
        if manifest_path is None or not manifest_path.exists():
            return timing
        try:
            data = json.loads(manifest_path.read_text())
            if not isinstance(data, dict):
                return timing
        except (json.JSONDecodeError, FileNotFoundError, PermissionError):
            return timing
        data["run_timing"] = timing
        if run_log_path is not None:
            data["run_log"] = str(run_log_path)
        manifest_path.write_text(json.dumps(data, indent=2))
        _record_artifact(manifest_path)
        return timing

    def _emit_run_end() -> None:
        nonlocal run_end_emitted
        if run_end_emitted:
            return
        timing = _finalize_manifest_timing()
        if run_log_path is None:
            run_end_emitted = True
            return
        run_end_emitted = True
        from .logging_utils import emit_run_end

        emit_run_end(
            duration_seconds=timing.get("duration_seconds", _current_duration()),
            started_at=timing.get("started_at"),
            ended_at=timing.get("ended_at"),
            seed=args.seed,
            backend=run_backend,
            artifact_paths=_collect_artifacts(),
            run_id=run_id,
            run_log=run_log_path,
            manifest_path=manifest_path,
        )

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
    from .facade import RunOptions, apply_run_options

    # Translate CLI flags into RunOptions before delegating execution to the facade.
    # The facade owns the simulation pipeline; the CLI owns argument parsing and output order.
    cfg = load_config(args.config)
    run_options = RunOptions(
        seed=args.seed,
        backend=args.backend,
        legacy_agent_rng=args.legacy_agent_rng,
        return_distribution=args.return_distribution,
        return_t_df=args.return_t_df,
        return_copula=args.return_copula,
        covariance_shrinkage=args.cov_shrinkage,
        vol_regime=args.vol_regime,
        vol_regime_window=args.vol_regime_window,
        analysis_mode=args.mode,
    )
    cfg = apply_run_options(cfg, run_options)
    # Resolve and set backend once, with proper signature
    backend_choice = resolve_and_set_backend(args.backend, cfg)
    args.backend = backend_choice
    run_backend = backend_choice
    run_options.backend = backend_choice

    # Echo backend selection at start (asserted in tests/expected_cli_outputs.py).
    print(f"[BACKEND] Using backend: {backend_choice}")

    from .data import load_index_returns
    from .facade import run_single
    from .logging_utils import setup_json_logging
    from .manifest import ManifestWriter
    from .reporting.attribution import (
        compute_sleeve_cvar_contribution,
        compute_sleeve_return_attribution,
        compute_sleeve_return_contribution,
        compute_sleeve_risk_attribution,
    )
    from .reporting.sweep_excel import export_sweep_results
    from .run_flags import RunFlags
    from .sim.simulation_initialization import initialize_sweep_rngs
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
    if args.log_json:
        # Create run directory under ./runs/<timestamp>
        ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = ts
        run_dir = Path("runs") / ts
        candidate_log_path = run_dir / "run.log"
        try:
            setup_json_logging(str(candidate_log_path), run_id=run_id)
            run_log_path = candidate_log_path
            _record_artifact(run_log_path)
        except (OSError, PermissionError, RuntimeError, ValueError) as e:
            logger.warning(f"Failed to set up JSON logging: {e}")

    base_cfg = cfg
    if args.stress_preset:
        base_cfg = cfg
        cfg = apply_stress_preset(cfg, args.stress_preset)

    idx_series = load_index_returns(args.index)

    # Ensure idx_series is a pandas Series for type safety
    if isinstance(idx_series, pd.DataFrame):
        idx_series = idx_series.squeeze()
        if not isinstance(idx_series, pd.Series):
            raise ValueError("Index data must be convertible to pandas Series")
    elif not isinstance(idx_series, pd.Series):
        raise ValueError("Index data must be a pandas Series")

    idx_series.attrs["source_path"] = str(args.index)

    # Handle frequency validation and resampling
    from .data.loaders import (
        FrequencyValidationError,
        resample_to_monthly,
        validate_frequency,
    )

    # If user explicitly declared frequency, store it in attrs
    if args.index_frequency:
        idx_series.attrs["frequency"] = args.index_frequency

    # Resample if requested
    if args.resample:
        idx_series = resample_to_monthly(idx_series)

    # Validate frequency (defaults to monthly, raises if mismatch)
    try:
        validate_frequency(idx_series, expected="monthly", strict=True)
    except FrequencyValidationError as e:
        raise SystemExit(f"Error: {e}") from None

    from .units import get_index_series_unit, normalize_index_series

    idx_series = normalize_index_series(idx_series, get_index_series_unit())
    index_hash = _hash_index_series(idx_series) if args.bundle else ""

    if args.register:
        try:
            from .scenario_registry import register as register_scenario

            scenario_id = register_scenario(cfg, idx_series, args.seed)
            print(f"[REGISTRY] Scenario ID: {scenario_id}")
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning(f"Failed to register scenario: {exc}")

    # Handle sleeve suggestion if requested
    if args.suggest_sleeves:
        suggest_seed = args.seed
        suggestions = suggest_sleeve_sizes(
            cfg,
            idx_series,
            max_te=args.max_te,
            max_breach=args.max_breach,
            max_cvar=args.max_cvar,
            max_shortfall=args.max_shortfall,
            step=args.sleeve_step,
            min_external=args.min_external,
            max_external=args.max_external,
            min_active=args.min_active,
            max_active=args.max_active,
            min_internal=args.min_internal,
            max_internal=args.max_internal,
            seed=suggest_seed,
            optimize=args.optimize,
            objective=args.optimize_objective,
        )
        if suggestions.empty:
            print("No feasible sleeve allocations found.")
            _emit_run_end()
            return
        if "optimizer_status" in suggestions.columns:
            status = str(suggestions.loc[0, "optimizer_status"])
            if status.startswith("grid_fallback"):
                print(f"Optimizer unavailable; using grid fallback ({status}).")
            elif status.startswith("fallback_failed"):
                print(f"Optimizer failed without grid fallback ({status}).")
        print(suggestions.to_string(index=True))
        idx_sel = args.suggest_apply_index
        if idx_sel is None:
            try:
                choice = input("Select row index to apply and continue (blank to abort): ").strip()
            except EOFError:
                print("No selection provided. Aborting run.")
                _emit_run_end()
                return
            if not choice:
                print("Aborting run.")
                _emit_run_end()
                return
            try:
                idx_sel = int(choice)
            except ValueError:
                print("Invalid selection. Aborting run.")
                _emit_run_end()
                return
        if idx_sel < 0 or idx_sel >= len(suggestions):
            print("Invalid selection. Aborting run.")
            _emit_run_end()
            return
        row = suggestions.iloc[idx_sel]
        cfg = cfg.model_copy(
            update={
                # Direct float conversion for clarity and efficiency
                "external_pa_capital": float(row["external_pa_capital"]),
                "active_ext_capital": float(row["active_ext_capital"]),
                "internal_pa_capital": float(row["internal_pa_capital"]),
            }
        )
        if args.stress_preset:
            base_cfg = base_cfg.model_copy(
                update={
                    "external_pa_capital": float(row["external_pa_capital"]),
                    "active_ext_capital": float(row["active_ext_capital"]),
                    "internal_pa_capital": float(row["internal_pa_capital"]),
                }
            )

    # Capture raw params after user-driven config adjustments (mode/stress/suggestions)
    raw_params = cfg.model_dump()

    substream_ids: Mapping[str, str] | None = None

    # Delegation: parameter sweep mode routes to run_parameter_sweep + export.
    if (
        cfg.analysis_mode in ["capital", "returns", "alpha_shares", "vol_mult"]
        and not args.sensitivity
    ):
        # Parameter sweep mode
        fin_agent_names = ["internal", "external_pa", "active_ext"]
        rng_bundle = initialize_sweep_rngs(
            args.seed,
            legacy_agent_rng=args.legacy_agent_rng,
            financing_agents=fin_agent_names,
        )
        rng_returns = rng_bundle.rng_returns
        fin_rngs = rng_bundle.rngs_financing
        substream_ids = rng_bundle.substream_ids
        results = run_parameter_sweep(
            cfg,
            idx_series,
            rng_returns,
            fin_rngs,
            seed=rng_bundle.seed,
        )
        sweep_metadata = {"rng_seed": args.seed, "substream_ids": substream_ids}
        export_sweep_results(results, filename=args.output, metadata=sweep_metadata)
        _record_artifact(args.output)

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
            substream_ids=substream_ids,
            cli_args=vars(args),
            backend=args.backend,
            run_log=run_log_path,
            previous_run=args.prev_manifest,
            run_timing=run_timer.snapshot(),
        )
        manifest_json = Path(args.output).with_name("manifest.json")
        manifest_path = manifest_json
        _record_artifact(manifest_json)
        manifest_data = None
        try:
            if manifest_json.exists():
                manifest_data = json.loads(manifest_json.read_text())
        except (json.JSONDecodeError, FileNotFoundError, PermissionError):
            manifest_data = None

        summary_frames = []
        for res in results:
            summary = res["summary"].copy()
            summary["terminal_ShortfallProb"] = summary.get("terminal_ShortfallProb", 0.0)
            summary["Combination"] = f"Run{res['combination_id']}"
            summary_frames.append(summary)
        all_summary = (
            pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        )

        current_manifest_data = manifest_data or {"config": raw_params}
        _maybe_print_run_diff(
            current_manifest=current_manifest_data,
            prev_manifest=prev_manifest_data,
            current_summary=all_summary,
            prev_summary=prev_summary_df,
        )

        # Handle packet export for parameter sweep mode
        if flags.packet:
            try:
                from . import viz

                if not all_summary.empty:
                    from .reporting.export_packet import (
                        create_export_packet as create_export_packet_fn,
                    )

                    # Create visualization from consolidated summary
                    if "terminal_ShortfallProb" in all_summary.columns:
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
                        prev_summary_df=prev_summary_df,
                        prev_manifest=prev_manifest_data,
                    )
                    _record_artifact(pptx_path)
                    _record_artifact(excel_path)
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
                print("💡 Install required packages: pip install plotly kaleido openpyxl")
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
                sweep_df = pd.concat([res["summary"] for res in results], ignore_index=True)
                base_agents = sweep_df[sweep_df["Agent"] == "Base"]
                if not base_agents.empty and isinstance(base_agents, pd.DataFrame):
                    best_combo = base_agents.loc[base_agents["terminal_AnnReturn"].idxmax()]
                    worst_combo = base_agents.loc[base_agents["terminal_AnnReturn"].idxmin()]
                    print(
                        f"   📈 Best combination: {best_combo['terminal_AnnReturn']:.2f}% terminal_AnnReturn"
                    )
                    print(
                        f"   📉 Worst combination: {worst_combo['terminal_AnnReturn']:.2f}% terminal_AnnReturn"
                    )
                    print(
                        f"   📊 Range: {best_combo['terminal_AnnReturn'] - worst_combo['terminal_AnnReturn']:.2f}% difference"
                    )
                else:
                    print("   ⚠️  No Base agent results found in sweep")
            else:
                print("   ❌ No sweep results available")

        _maybe_write_bundle(index_hash=index_hash, manifest_data=manifest_data)
        _emit_run_end()
        return

    # Normal single-run mode below delegates to run_single + optional exports.
    run_artifacts = run_single(cfg, idx_series, run_options)
    returns = run_artifacts.returns
    summary = run_artifacts.summary
    run_manifest: dict[str, Any] = dict(run_artifacts.manifest or {})
    substream_ids = cast(dict[str, str] | None, run_manifest.get("substream_ids"))
    stress_delta_df = None
    base_summary_df: pd.DataFrame | None = None
    if args.stress_preset:
        from .reporting.stress_delta import build_delta_table

        base_artifacts = run_single(base_cfg, idx_series, run_options)
        base_summary_df = base_artifacts.summary
        stress_delta_df = build_delta_table(base_summary_df, summary)
    inputs_dict: dict[str, object] = {k: raw_params.get(k, "") for k in raw_params}
    for key in ("correlation_repair_applied", "correlation_repair_details"):
        if key in run_artifacts.inputs:
            inputs_dict[key] = run_artifacts.inputs[key]
    raw_returns_dict = dict(run_artifacts.raw_returns)

    # Optional attribution tables for downstream exports
    try:
        inputs_dict["_attribution_df"] = compute_sleeve_return_attribution(cfg, idx_series)
    except (AttributeError, TypeError):  # narrow exceptions required by tests
        # Fallback: aggregate total annualised return by agent if detailed attribution fails
        try:
            rows: list[dict[str, object]] = []
            for agent, arr in returns.items():
                mean_month = float(np.asarray(arr).mean())
                ann = 12.0 * mean_month
                rows.append({"Agent": agent, "Sub": "Total", "Return": ann})
            inputs_dict["_attribution_df"] = pd.DataFrame(rows)
        except (AttributeError, ValueError, TypeError, KeyError) as e2:
            logger.debug(f"Attribution fallback unavailable: {e2}")
            inputs_dict["_attribution_df"] = pd.DataFrame(
                [{"Agent": "", "Sub": "", "Return": 0.0}]
            ).head(0)
    try:
        sleeve_attr = compute_sleeve_return_contribution(returns)
        cvar_attr = compute_sleeve_cvar_contribution(returns)
        if not cvar_attr.empty:
            sleeve_attr = sleeve_attr.merge(cvar_attr, on="Agent", how="outer")
        inputs_dict["_sleeve_attribution_df"] = sleeve_attr
    except (AttributeError, ValueError, TypeError, KeyError) as e:
        logger.debug(f"Sleeve attribution unavailable: {e}")
    try:
        inputs_dict["_risk_attr_df"] = compute_sleeve_risk_attribution(cfg, idx_series)
    except (AttributeError, ValueError, TypeError, KeyError) as e:
        logger.debug(f"Risk attribution unavailable: {e}")
    print_enhanced_summary(summary)
    base_constraints_report: pd.DataFrame | None = None
    stress_constraints_report: pd.DataFrame | None = None
    try:
        from .reporting.console import print_constraint_report
        from .reporting.constraints import build_constraint_report

        constraint_report = build_constraint_report(
            summary,
            max_te=args.max_te,
            max_breach=args.max_breach,
            max_cvar=args.max_cvar,
        )
        inputs_dict["_constraint_report_df"] = constraint_report
        if args.stress_preset:
            stress_constraints_report = constraint_report
            if base_summary_df is not None and not base_summary_df.empty:
                base_constraints_report = build_constraint_report(
                    base_summary_df,
                    max_te=args.max_te,
                    max_breach=args.max_breach,
                    max_cvar=args.max_cvar,
                )
            print_constraint_report(constraint_report)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        logger.debug(f"Constraint report unavailable: {e}")
    # Optional: compute trade-off table (non-interactive) and attach for export
    if args.tradeoff_table:
        try:
            suggest_seed = args.seed
            trade_df = suggest_sleeve_sizes(
                cfg,
                idx_series,
                max_te=args.max_te,
                max_breach=args.max_breach,
                max_cvar=args.max_cvar,
                max_shortfall=args.max_shortfall,
                step=args.sleeve_step,
                min_external=args.min_external,
                max_external=args.max_external,
                min_active=args.min_active,
                max_active=args.max_active,
                min_internal=args.min_internal,
                max_internal=args.max_internal,
                seed=suggest_seed,
                sort_by=args.tradeoff_sort,
            )
            if not trade_df.empty:
                inputs_dict["_tradeoff_df"] = trade_df.head(max(1, args.tradeoff_top)).reset_index(
                    drop=True
                )
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
    # Optional sensitivity analysis (one-factor deltas on terminal_AnnReturn)
    if args.sensitivity:
        try:
            from .sensitivity import one_factor_deltas as simple_one_factor_deltas

            print("\n🔍 Running sensitivity analysis...")

            # Build a simple evaluator: change a single param, re-run summary terminal_AnnReturn for Base
            def _eval(p: dict[str, float]) -> float:
                """Evaluate terminal_AnnReturn for Base agent given parameter overrides."""
                mod_cfg = cfg.model_copy(update=p)
                mod_artifacts = run_single(mod_cfg, idx_series, run_options)
                summary_l = mod_artifacts.summary
                base_row = summary_l[summary_l["Agent"] == "Base"]
                if isinstance(base_row, pd.DataFrame) and not base_row.empty:
                    return float(base_row["terminal_AnnReturn"].iloc[0])
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
            param_results: dict[str, dict[str, float | None]] = {
                name: {"plus": None, "minus": None} for name in base_params
            }

            for param_name, base_value in base_params.items():
                # Test positive perturbation
                pos_key = f"{param_name}_+5%"
                try:
                    pos_value = base_value * 1.05
                    pos_result = _eval({param_name: pos_value})
                    scenarios[pos_key] = pd.DataFrame({"terminal_AnnReturn": [pos_result]})
                    param_results[param_name]["plus"] = pos_result
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
                    scenarios[neg_key] = pd.DataFrame({"terminal_AnnReturn": [neg_result]})
                    param_results[param_name]["minus"] = neg_result
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
                base_df = summary[summary["Agent"] == "Base"][["terminal_AnnReturn"]]
                if not isinstance(base_df, pd.DataFrame):
                    base_df = pd.DataFrame(base_df)
                deltas = simple_one_factor_deltas(base_df, scenarios, value="terminal_AnnReturn")
                base_value = (
                    float(base_df["terminal_AnnReturn"].iloc[0]) if not base_df.empty else 0.0
                )
                records = []
                for name, values in param_results.items():
                    minus_val = values.get("minus")
                    plus_val = values.get("plus")
                    if minus_val is None or plus_val is None:
                        continue
                    low = minus_val - base_value
                    high = plus_val - base_value
                    delta_abs = max(abs(low), abs(high))
                    records.append((name, base_value, minus_val, plus_val, low, high, delta_abs))
                if records:
                    sens_df = pd.DataFrame(
                        records,
                        columns=[
                            "Parameter",
                            "Base",
                            "Minus",
                            "Plus",
                            "Low",
                            "High",
                            "DeltaAbs",
                        ],
                    )
                    sens_df.sort_values(
                        ["DeltaAbs", "Parameter"],
                        ascending=[False, True],
                        inplace=True,
                        kind="mergesort",
                    )
                    sens_df.reset_index(drop=True, inplace=True)
                    sens_df.attrs.update(
                        {
                            "metric": "terminal_AnnReturn",
                            "units": "%",
                            "tickformat": ".2%",
                        }
                    )
                    inputs_dict["_sensitivity_df"] = sens_df

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
                    print("\n💡 Consider reviewing parameter ranges or model constraints.")

                print(f"\n✅ Sensitivity analysis completed. Evaluated {len(scenarios)} scenarios.")
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

    metadata = {"rng_seed": run_manifest.get("seed", args.seed), "substream_ids": substream_ids}
    finalize_after_append = bool(args.stress_preset)
    deps.export_to_excel(
        inputs_dict,
        summary,
        raw_returns_dict,
        filename=flags.save_xlsx or "Outputs.xlsx",
        pivot=args.pivot,
        finalize=not finalize_after_append,
        metadata=metadata,
    )
    _record_artifact(flags.save_xlsx or "Outputs.xlsx")
    if args.stress_preset:
        out_path = Path(flags.save_xlsx or "Outputs.xlsx")
        if out_path.exists():
            try:
                with pd.ExcelWriter(
                    out_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
                ) as writer:
                    if base_summary_df is not None and not base_summary_df.empty:
                        base_summary_df.to_excel(writer, sheet_name="BaseSummary", index=False)
                    if summary is not None and not summary.empty:
                        summary.to_excel(writer, sheet_name="StressedSummary", index=False)
                    if stress_delta_df is not None and not stress_delta_df.empty:
                        stress_delta_df.to_excel(writer, sheet_name="StressDelta", index=False)
                    if base_constraints_report is not None and not base_constraints_report.empty:
                        base_constraints_report.to_excel(
                            writer, sheet_name="BaseBreaches", index=False
                        )
                    if (
                        stress_constraints_report is not None
                        and not stress_constraints_report.empty
                    ):
                        stress_constraints_report.to_excel(
                            writer, sheet_name="StressedBreaches", index=False
                        )
            except (OSError, PermissionError, ValueError) as e:
                logger.warning(f"Failed to append stress sheets: {e}")
            finally:
                try:
                    from .reporting.excel import finalize_excel_workbook

                    finalize_excel_workbook(str(out_path), inputs_dict, summary)
                except (OSError, PermissionError, ValueError) as e:
                    logger.warning(f"Failed to finalize stress workbook: {e}")
        else:
            logger.warning("Stress sheet export skipped; output workbook missing.")

    # Write reproducibility manifest for normal run
    try:
        mw = ManifestWriter(Path(flags.save_xlsx or "Outputs.xlsx").with_name("manifest.json"))
        data_files = [args.index, args.config]
        out_path = Path(flags.save_xlsx or "Outputs.xlsx")
        if out_path.exists():
            data_files.append(str(out_path))
        mw.write(
            config_path=args.config,
            data_files=data_files,
            seed=args.seed,
            substream_ids=substream_ids,
            cli_args=vars(args),
            backend=args.backend,
            run_log=run_log_path,
            previous_run=args.prev_manifest,
            run_timing=run_timer.snapshot(),
        )
        manifest_path = Path(flags.save_xlsx or "Outputs.xlsx").with_name("manifest.json")
    except (OSError, PermissionError, FileNotFoundError) as e:
        logger.warning(f"Failed to write manifest: {e}")

    manifest_data = None
    try:
        manifest_json = Path(flags.save_xlsx or "Outputs.xlsx").with_name("manifest.json")
        _record_artifact(manifest_json)
        if manifest_json.exists():
            manifest_data = json.loads(manifest_json.read_text())
    except (json.JSONDecodeError, FileNotFoundError, PermissionError):
        manifest_data = None

    current_manifest_data = manifest_data or {"config": raw_params}
    _maybe_print_run_diff(
        current_manifest=current_manifest_data,
        prev_manifest=prev_manifest_data,
        current_summary=summary,
        prev_summary=prev_summary_df,
    )

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
        from . import viz

        plots = Path("plots")
        plots.mkdir(exist_ok=True)
        # Guard summary type for static checkers
        if isinstance(summary, pd.DataFrame) and ("terminal_ShortfallProb" in summary.columns):
            fig = viz.risk_return.make(summary)
        else:
            fig = viz.sharpe_ladder.make(summary)
        stem = plots / "summary"
        if flags.png:
            _record_artifact(stem.with_suffix(".png"))
        if flags.pdf:
            _record_artifact(stem.with_suffix(".pdf"))
        if flags.pptx:
            _record_artifact(stem.with_suffix(".pptx"))
        if flags.html:
            _record_artifact(stem.with_suffix(".html"))
        if flags.gif:
            _record_artifact(plots / "paths.gif")

        # Handle packet export first (comprehensive export)
        if flags.packet:
            try:
                from . import viz

                # Use base filename from --output or default
                base_name = Path(flags.save_xlsx or "committee_packet").stem
                # Load manifest (if written) for embedding
                manifest_json = Path(flags.save_xlsx or "Outputs.xlsx").with_name("manifest.json")
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
                            series = viz.tornado.series_from_sensitivity(sens_df_plot)
                            figs.append(viz.tornado.make(series, title="Sensitivity Tornado"))
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
                    logger.debug("Skipping sunburst figure due to data issue", exc_info=e)
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
                    stress_delta_df=stress_delta_df,
                )
                _record_artifact(pptx_path)
                _record_artifact(excel_path)
                print("✅ Export packet created:")
                print(f"   📊 Excel: {excel_path}")
                print(f"   📋 PowerPoint: {pptx_path}")
            except RuntimeError as e:
                print(f"❌ Export packet failed: {e}")
                _emit_run_end()
                return
            except (ImportError, ModuleNotFoundError) as e:
                logger.error(f"Export packet failed due to missing dependency: {e}")
                print(f"❌ Export packet failed due to missing dependency: {e}")
                print(
                    "💡 Install required packages: pip install plotly kaleido openpyxl python-pptx"
                )
                _emit_run_end()
                return
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Export packet failed due to data/config issue: {e}")
                print(f"❌ Export packet failed due to data or configuration issue: {e}")
                print("💡 Check your data inputs and configuration settings")
                _emit_run_end()
                return
            except (OSError, PermissionError) as e:
                logger.error(f"Export packet failed due to file system issue: {e}")
                print(f"❌ Export packet failed due to file system issue: {e}")
                print("💡 Check file permissions and available disk space")
                _emit_run_end()
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
                if any(term in msg for term in ("kaleido", "chrome", "chromium", "cancelled")):
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
                pptx_figs = [fig]
                sens_val = inputs_dict.get("_sensitivity_df")
                sens_df = sens_val if isinstance(sens_val, pd.DataFrame) else None
                if sens_df is not None and not sens_df.empty:
                    if {"Parameter", "DeltaAbs"} <= set(sens_df.columns):
                        series = viz.tornado.series_from_sensitivity(sens_df)
                        pptx_figs.append(viz.tornado.make(series, title="Sensitivity Tornado"))
                viz.pptx_export.save(
                    pptx_figs,
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
                _emit_run_end()
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
                    raise FileNotFoundError(f"Dashboard file not found: {dashboard_path}")

                subprocess.run(
                    [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
                    check=True,
                    cwd=os.getcwd(),
                )
            except FileNotFoundError as e:
                print(f"❌ Dashboard launch failed: {e}")
                print("💡 Ensure the dashboard files are present in the 'dashboard/' directory.")
                _emit_run_end()
                return
            except subprocess.CalledProcessError as e:
                logger.error(f"Dashboard launch failed with exit code {e.returncode}: {e}")
                print(f"❌ Dashboard launch failed with exit code {e.returncode}")
                print("💡 Common solutions:")
                print("   • Install Streamlit: pip install streamlit")
                print("   • Check if 'dashboard/app.py' is valid Python code")
                print("   • Verify your Python environment is properly configured")
                _emit_run_end()
                return
            except ImportError as e:
                logger.error(f"Dashboard launch failed due to missing streamlit: {e}")
                print(f"❌ Dashboard launch failed: Streamlit not available - {e}")
                print("💡 Install Streamlit: pip install streamlit")
                _emit_run_end()
                return
            except (OSError, PermissionError) as e:
                logger.error(f"Dashboard launch failed due to system issue: {e}")
                print(f"❌ Dashboard launch failed: System/permission error - {e}")
                print("💡 Check file permissions and system resources")
                _emit_run_end()
                return

    _maybe_write_bundle(index_hash=index_hash, manifest_data=manifest_data)
    _emit_run_end()


# (Backward compatibility global variable assignment removed)
if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
