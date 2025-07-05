"""Command-line interface for running simulations.

Additional options allow exporting visualisations and launching the
Streamlit dashboard after a run.

CLI flags:
    --png / --pdf / --pptx  Static exports (can be combined)
    --html                 Save interactive HTML
    --gif                  Animated export of monthly paths
    --dashboard            Launch Streamlit dashboard after run
"""
from __future__ import annotations

import argparse
from typing import Sequence, Optional
import pandas as pd

from . import (
    load_parameters,
    load_index_returns,
    draw_joint_returns,
    draw_financing_series,
    export_to_excel,
    print_summary,
    load_config,
)
from .sim.covariance import build_cov_matrix
from .backend import set_backend
from .random import spawn_rngs, spawn_agent_rngs
from .agents.registry import build_from_config
from .simulations import simulate_agents
from .sim.metrics import summary_table

LABEL_MAP = {
    "Analysis mode": "analysis_mode",
    "Number of simulations": "N_SIMULATIONS",
    "Number of months": "N_MONTHS",
    "External PA capital (mm)": "external_pa_capital",
    "Active Extension capital (mm)": "active_ext_capital",
    "Internal PA capital (mm)": "internal_pa_capital",
    "In-House annual return (%)": "mu_H",
    "In-House annual vol (%)": "sigma_H",
    "Alpha-Extension annual return (%)": "mu_E",
    "Alpha-Extension annual vol (%)": "sigma_E",
    "External annual return (%)": "mu_M",
    "External annual vol (%)": "sigma_M",
    "Corr index–In-House": "rho_idx_H",
    "Corr index–Alpha-Extension": "rho_idx_E",
    "Corr index–External": "rho_idx_M",
    "Corr In-House–Alpha-Extension": "rho_H_E",
    "Corr In-House–External": "rho_H_M",
    "Corr Alpha-Extension–External": "rho_E_M",
    "Total fund capital (mm)": "total_fund_capital",
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Portable Alpha simulation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--params", help="Parameters CSV")
    group.add_argument("--config", help="YAML config file")
    parser.add_argument("--index", required=True, help="Index returns CSV")
    parser.add_argument("--output", default="Outputs.xlsx", help="Output workbook")
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
        "--dashboard",
        action="store_true",
        help="Launch Streamlit dashboard after run",
    )
    args = parser.parse_args(argv)

    set_backend(args.backend)

    rng_returns = spawn_rngs(args.seed, 1)[0]
    fin_rngs = spawn_agent_rngs(
        args.seed,
        ["internal", "external_pa", "active_ext"],
    )

    if args.config:
        cfg = load_config(args.config)
    else:
        raw_params = load_parameters(args.params, LABEL_MAP)
        cfg = load_config(raw_params)
    raw_params = cfg.dict()
    idx_series = load_index_returns(args.index)
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

    summary = summary_table(returns, benchmark="Base")
    inputs_dict = {k: raw_params.get(k, "") for k in raw_params}
    raw_returns_dict = {k: pd.DataFrame(v) for k, v in returns.items()}
    print_summary(summary)
    export_to_excel(
        inputs_dict,
        summary,
        raw_returns_dict,
        filename=args.output,
        pivot=args.pivot,
    )

    if any([args.png, args.pdf, args.pptx, args.html, args.gif, args.dashboard]):
        from pathlib import Path
        from . import viz
        plots = Path("plots")
        plots.mkdir(exist_ok=True)
        if "ShortfallProb" in summary.columns:
            fig = viz.risk_return.make(summary)
        else:
            fig = viz.sharpe_ladder.make(summary)
        stem = plots / "summary"
        if args.png:
            try:
                fig.write_image(stem.with_suffix(".png"))
            except Exception:
                pass
        if args.pdf:
            try:
                viz.pdf_export.save(fig, str(stem.with_suffix(".pdf")))
            except Exception:
                pass
        if args.pptx:
            try:
                viz.pptx_export.save([fig], str(stem.with_suffix(".pptx")))
            except Exception:
                pass
        if args.html:
            viz.html_export.save(fig, str(stem.with_suffix(".html")))
        if args.gif:
            arr = next(iter(raw_returns_dict.values())).to_numpy()
            anim = viz.animation.make(arr)
            try:
                anim.write_image(str(plots / "paths.gif"))
            except Exception:
                pass
        if args.dashboard:
            import subprocess
            subprocess.run(["streamlit", "run", "dashboard/app.py"], check=False)


