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
from typing import Optional, Sequence, TYPE_CHECKING
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

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
from .stress import STRESS_PRESETS, apply_stress_preset
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
        "--packet",
        action="store_true",
        help="Export comprehensive committee packet (PPTX + Excel)",
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
        packet=args.packet,
    )

    set_backend(args.backend)

    rng_returns = spawn_rngs(args.seed, 1)[0]
    fin_rngs = spawn_agent_rngs(
        args.seed,
        ["internal", "external_pa", "active_ext"],
    )

    cfg = load_config(args.config)
    cfg = cfg.model_copy(update={"analysis_mode": args.mode})
    if args.stress_preset:
        cfg = apply_stress_preset(cfg, args.stress_preset)
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
        )
        
        # Handle packet export for parameter sweep mode
        if flags.packet:
            try:
                from .reporting.export_packet import create_export_packet
                from . import viz
                
                # Build consolidated summary from sweep results (similar to export_sweep_results)
                summary_frames = []
                for res in results:
                    summary = res["summary"].copy()
                    summary["ShortfallProb"] = summary.get("ShortfallProb", 0.0)
                    summary["Combination"] = f"Run{res['combination_id']}"
                    summary_frames.append(summary)
                
                if summary_frames:
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
                    
                    pptx_path, excel_path = create_export_packet(
                        figs=[fig],
                        summary_df=all_summary,
                        raw_returns_dict=raw_returns_dict,
                        inputs_dict={k: raw_params.get(k, "") for k in raw_params},
                        base_filename=base_name,
                        alt_texts=[flags.alt_text] if flags.alt_text else None,
                        pivot=args.pivot,
                    )
                    print("✅ Parameter sweep export packet created:")
                    print(f"   📊 Excel: {excel_path}")
                    print(f"   📋 PowerPoint: {pptx_path}")
                else:
                    print("⚠️  No summary data available for export packet")
            except RuntimeError as e:
                print(f"❌ Export packet failed: {e}")
            except Exception as e:
                print(f"❌ Unexpected error creating export packet: {e}")
                print("💡 Please check your environment and try individual exports instead.")
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

    if any([flags.png, flags.pdf, flags.pptx, flags.html, flags.gif, flags.dashboard, flags.packet]):
        pass

    if any([flags.png, flags.pdf, flags.pptx, flags.html, flags.gif, flags.dashboard]):
        from . import viz

        plots = Path("plots")
        plots.mkdir(exist_ok=True)
        if "ShortfallProb" in summary.columns:
            fig = viz.risk_return.make(summary)
        else:
            fig = viz.sharpe_ladder.make(summary)
        stem = plots / "summary"
        
        # Handle packet export first (comprehensive export)
        if flags.packet:
            try:
                from .reporting.export_packet import create_export_packet
                
                # Use base filename from --output or default
                base_name = Path(flags.save_xlsx or "committee_packet").stem
                pptx_path, excel_path = create_export_packet(
                    figs=[fig],
                    summary_df=summary,
                    raw_returns_dict=raw_returns_dict,
                    inputs_dict=inputs_dict,
                    base_filename=base_name,
                    alt_texts=[flags.alt_text] if flags.alt_text else None,
                    pivot=args.pivot,
                )
                print("✅ Export packet created:")
                print(f"   📊 Excel: {excel_path}")
                print(f"   📋 PowerPoint: {pptx_path}")
            except RuntimeError as e:
                print(f"❌ Export packet failed: {e}")
                return
            except Exception as e:
                print(f"❌ Unexpected error creating export packet: {e}")
                print("💡 Please check your environment and try individual exports instead.")
                return
        
        # Individual export formats (with improved error handling)
        if flags.png:
            try:
                fig.write_image(stem.with_suffix(".png"))
            except Exception as e:
                if "Chrome" in str(e) or "Kaleido" in str(e) or "Chromium" in str(e):
                    print("❌ PNG export failed: Chrome/Chromium required")
                    print("💡 Install with: sudo apt-get install chromium-browser")
                else:
                    print(f"❌ PNG export failed: {e}")
        if flags.pdf:
            try:
                viz.pdf_export.save(fig, str(stem.with_suffix(".pdf")))
            except Exception as e:
                if "Chrome" in str(e) or "Kaleido" in str(e) or "Chromium" in str(e):
                    print("❌ PDF export failed: Chrome/Chromium required")
                    print("💡 Install with: sudo apt-get install chromium-browser")
                else:
                    print(f"❌ PDF export failed: {e}")
        if flags.pptx:
            try:
                viz.pptx_export.save(
                    [fig],
                    str(stem.with_suffix(".pptx")),
                    alt_texts=[flags.alt_text] if flags.alt_text else None,
                )
            except Exception as e:
                if "Chrome" in str(e) or "Kaleido" in str(e) or "Chromium" in str(e):
                    print("❌ PPTX export failed: Chrome/Chromium required")
                    print("💡 Install with: sudo apt-get install chromium-browser")
                else:
                    print(f"❌ PPTX export failed: {e}")
        if flags.html:
            viz.html_export.save(
                fig,
                str(stem.with_suffix(".html")),
                alt_text=flags.alt_text,
            )
        if flags.gif:
            try:
                arr = next(iter(raw_returns_dict.values())).to_numpy()
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
