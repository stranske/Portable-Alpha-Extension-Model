"""Command-line interface for running simulations.

Additional options allow exporting visualisations and launching the
Streamlit dashboard after a run.

CLI flags:
    --png / --pdf / --pptx  Static exports (can be combined)
    --html                 Save interactive HTML
    --gif                  Animated export of monthly paths
    --alt-text TEXT        Alt text for HTML/PPTX exports
    --dashboard            Launch Streamlit dashboard after run
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, TYPE_CHECKING
from pathlib import Path

import pandas as pd

if TYPE_CHECKING:
    import numpy as np

from . import (
    RunFlags,
    draw_financing_series,
    draw_joint_returns,
    export_to_excel,
    load_config,
    load_index_returns,
)
from .agents.registry import build_from_config
from .backend import set_backend
from .random import spawn_agent_rngs, spawn_rngs
from .reporting.console import print_summary
from .reporting.sweep_excel import export_sweep_results
from .sim.covariance import build_cov_matrix
from .sim.metrics import summary_table
from .simulations import simulate_agents
from .sweep import run_parameter_sweep
from .manifest import ManifestWriter



def create_enhanced_summary(
    returns_map: dict[str, np.ndarray],
    *,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Create summary table with standard breach and shortfall defaults."""

    return summary_table(returns_map, benchmark=benchmark)


def print_enhanced_summary(summary: pd.DataFrame) -> None:
    """Print enhanced summary with explanations."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

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


def main(argv: Optional[Sequence[str]] = None) -> None:
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
        "--pivot",
        action="store_true",
        help="Write all raw returns in a single long-format sheet",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "cupy"],
        default="numpy",
        help="Computation backend",
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
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard after run",
    )
    args = parser.parse_args(argv)

    flags = RunFlags(
        save_xlsx=args.output,
        png=args.png,
        pdf=args.pdf,
        pptx=args.pptx,
        html=args.html,
        gif=args.gif,
        dashboard=args.dashboard,
        alt_text=args.alt_text,
    )

    set_backend(args.backend)

    rng_returns = spawn_rngs(args.seed, 1)[0]
    fin_rngs = spawn_agent_rngs(
        args.seed,
        ["internal", "external_pa", "active_ext"],
    )

    cfg = load_config(args.config)
    cfg = cfg.model_copy(update={"analysis_mode": args.mode})
    raw_params = cfg.model_dump()
    idx_series = load_index_returns(args.index)

    # Ensure idx_series is a pandas Series for type safety
    if isinstance(idx_series, pd.DataFrame):
        idx_series = idx_series.squeeze()
        if not isinstance(idx_series, pd.Series):
            raise ValueError("Index data must be convertible to pandas Series")
    elif not isinstance(idx_series, pd.Series):
        raise ValueError("Index data must be a pandas Series")

    if cfg.analysis_mode in ["capital", "returns", "alpha_shares", "vol_mult"]:
        # Parameter sweep mode
        results = run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)
        export_sweep_results(results, filename=args.output)
        manifest_path = Path(args.output).with_name("manifest.json")
        ManifestWriter(manifest_path).write(
            config_path=args.config,
            data_files=[args.index, args.config],
            seed=args.seed,
            cli_args=vars(args),
        )
        return
    mu_idx = float(idx_series.mean())
    idx_sigma = float(idx_series.std(ddof=1))

    mu_H = cfg.mu_H
    sigma_H = cfg.sigma_H
    mu_E = cfg.mu_E
    sigma_E = cfg.sigma_E
    mu_M = cfg.mu_M
    sigma_M = cfg.sigma_M

    _ = build_cov_matrix(
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

    r_beta, r_H, r_E, r_M = draw_joint_returns(
        n_months=N_MONTHS,
        n_sim=N_SIMULATIONS,
        params=params,
        rng=rng_returns,
    )
    f_int, f_ext, f_act = draw_financing_series(
        n_months=N_MONTHS,
        n_sim=N_SIMULATIONS,
        params=params,
        rngs=fin_rngs,
    )

    # Build agents based on the configuration
    agents = build_from_config(cfg)

    returns = simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act)

    # Enhanced summary with better defaults and ShortfallProb
    summary = create_enhanced_summary(returns, benchmark="Base")
    inputs_dict = {k: raw_params.get(k, "") for k in raw_params}
    raw_returns_dict = {k: pd.DataFrame(v) for k, v in returns.items()}
    print_enhanced_summary(summary)
    export_to_excel(
        inputs_dict,
        summary,
        raw_returns_dict,
        filename=flags.save_xlsx or "Outputs.xlsx",
        pivot=args.pivot,
    )

    # Write reproducibility manifest
    manifest_path = Path(flags.save_xlsx or "Outputs.xlsx").with_name("manifest.json")
    ManifestWriter(manifest_path).write(
        config_path=args.config,
        data_files=[args.index, args.config],
        seed=args.seed,
        cli_args=vars(args),
    )

    if any([flags.png, flags.pdf, flags.pptx, flags.html, flags.gif, flags.dashboard]):
        from . import viz

        plots = Path("plots")
        plots.mkdir(exist_ok=True)
        if "ShortfallProb" in summary.columns:
            fig = viz.risk_return.make(summary)
        else:
            fig = viz.sharpe_ladder.make(summary)
        stem = plots / "summary"
        if flags.png:
            try:
                fig.write_image(stem.with_suffix(".png"))
            except Exception:
                pass
        if flags.pdf:
            try:
                viz.pdf_export.save(fig, str(stem.with_suffix(".pdf")))
            except Exception:
                pass
        if flags.pptx:
            try:
                viz.pptx_export.save(
                    [fig],
                    str(stem.with_suffix(".pptx")),
                    alt_texts=[flags.alt_text] if flags.alt_text else None,
                )
            except Exception:
                pass
        if flags.html:
            viz.html_export.save(
                fig,
                str(stem.with_suffix(".html")),
                alt_text=flags.alt_text,
            )
        if flags.gif:
            arr = next(iter(raw_returns_dict.values())).to_numpy()
            anim = viz.animation.make(arr)
            try:
                anim.write_image(str(plots / "paths.gif"))
            except Exception:
                pass
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
                return
            except subprocess.CalledProcessError as e:
                print(f"❌ Dashboard launch failed with exit code {e.returncode}")
                print("💡 Common solutions:")
                print("   • Install Streamlit: pip install streamlit")
                print("   • Check if 'dashboard/app.py' is valid Python code")
                print("   • Verify your Python environment is properly configured")
                return
            except Exception as e:
                print(f"❌ Unexpected error launching dashboard: {e}")
                print("💡 Please check your Python environment and try running manually:")
                print(f"   {sys.executable} -m streamlit run dashboard/app.py")
                return


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
