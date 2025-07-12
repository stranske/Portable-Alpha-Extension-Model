from __future__ import annotations

import argparse
from typing import Optional, Sequence

import pandas as pd

from . import (
    draw_financing_series,
    draw_joint_returns,
    export_to_excel,
    load_config,
    load_index_returns,
    load_parameters,
)
from .agents.registry import build_from_config
from .backend import set_backend
from .random import spawn_agent_rngs, spawn_rngs
from .sim.covariance import build_cov_matrix
from .sim.metrics import (
    summary_table,
)
from .simulations import simulate_agents

LABEL_MAP = {
    "Analysis mode": "analysis_mode",
    "Number of simulations": "N_SIMULATIONS",
    "Number of months": "N_MONTHS",
    "External PA capital (mm)": "external_pa_capital",
    "Active Extension capital (mm)": "active_ext_capital",
    "Internal PA capital (mm)": "internal_pa_capital",
    "In-House beta share": "w_beta_H",
    "In-House alpha share": "w_alpha_H",
    "External PA alpha fraction": "theta_extpa",
    "Active share (%)": "active_share",
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
    "Internal financing mean (monthly %)": "internal_financing_mean_month",
    "Internal financing vol (monthly %)": "internal_financing_sigma_month",
    "Internal monthly spike prob": "internal_spike_prob",
    "Internal spike multiplier": "internal_spike_factor",
    "External PA financing mean (monthly %)": "ext_pa_financing_mean_month",
    "External PA financing vol (monthly %)": "ext_pa_financing_sigma_month",
    "External PA monthly spike prob": "ext_pa_spike_prob",
    "External PA spike multiplier": "ext_pa_spike_factor",
    "Active Ext financing mean (monthly %)": "act_ext_financing_mean_month",
    "Active Ext financing vol (monthly %)": "act_ext_financing_sigma_month",
    "Active Ext monthly spike prob": "act_ext_spike_prob",
    "Active Ext spike multiplier": "act_ext_spike_factor",
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

    # Build agents from configuration
    agents = build_from_config(cfg)

    returns = simulate_agents(agents, r_beta, r_H, r_E, r_M, f_int, f_ext, f_act)

    summary = summary_table(returns, benchmark="Base")
    inputs_dict = {k: raw_params.get(k, "") for k in raw_params}
    raw_returns_dict = {k: pd.DataFrame(v) for k, v in returns.items()}
    export_to_excel(inputs_dict, summary, raw_returns_dict, filename=args.output)
