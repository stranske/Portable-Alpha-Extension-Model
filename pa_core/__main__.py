from __future__ import annotations
import argparse
from typing import Sequence, Optional
import pandas as pd

from . import (
    load_parameters,
    get_num,
    load_index_returns,
    draw_joint_returns,
    draw_financing_series,
    export_to_excel,
    load_config,
)
from .covariance import build_cov_matrix

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
    args = parser.parse_args(argv)

    if args.config:
        cfg = load_config(args.config)
        raw_params = cfg.dict()
    else:
        raw_params = load_parameters(args.params, LABEL_MAP)
    idx_series = load_index_returns(args.index)
    mu_idx = float(idx_series.mean())
    idx_sigma = float(idx_series.std(ddof=1))

    mu_H = get_num(raw_params, "mu_H", 0.04)
    sigma_H = get_num(raw_params, "sigma_H", 0.01)
    mu_E = get_num(raw_params, "mu_E", 0.05)
    sigma_E = get_num(raw_params, "sigma_E", 0.02)
    mu_M = get_num(raw_params, "mu_M", 0.03)
    sigma_M = get_num(raw_params, "sigma_M", 0.02)

    _ = build_cov_matrix(
        get_num(raw_params, "rho_idx_H", 0.05),
        get_num(raw_params, "rho_idx_E", 0.0),
        get_num(raw_params, "rho_idx_M", 0.0),
        get_num(raw_params, "rho_H_E", 0.10),
        get_num(raw_params, "rho_H_M", 0.10),
        get_num(raw_params, "rho_E_M", 0.0),
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
        "rho_idx_H": get_num(raw_params, "rho_idx_H", 0.05),
        "rho_idx_E": get_num(raw_params, "rho_idx_E", 0.0),
        "rho_idx_M": get_num(raw_params, "rho_idx_M", 0.0),
        "rho_H_E": get_num(raw_params, "rho_H_E", 0.10),
        "rho_H_M": get_num(raw_params, "rho_H_M", 0.10),
        "rho_E_M": get_num(raw_params, "rho_E_M", 0.0),
        "internal_financing_mean_month": 0.0,
        "internal_financing_sigma_month": 0.0,
        "internal_spike_prob": 0.0,
        "internal_spike_factor": 0.0,
        "ext_pa_financing_mean_month": 0.0,
        "ext_pa_financing_sigma_month": 0.0,
        "ext_pa_spike_prob": 0.0,
        "ext_pa_spike_factor": 0.0,
        "act_ext_financing_mean_month": 0.0,
        "act_ext_financing_sigma_month": 0.0,
        "act_ext_spike_prob": 0.0,
        "act_ext_spike_factor": 0.0,
    }

    N_SIMULATIONS = int(get_num(raw_params, "N_SIMULATIONS", 1000))
    N_MONTHS = int(get_num(raw_params, "N_MONTHS", 12))

    r_beta, r_H, r_E, r_M = draw_joint_returns(
        n_months=N_MONTHS,
        n_sim=N_SIMULATIONS,
        params=params,
    )
    f_int, f_ext, f_act = draw_financing_series(
        n_months=N_MONTHS,
        n_sim=N_SIMULATIONS,
        params=params,
    )

    base_returns = r_beta - f_int
    summary = pd.DataFrame({
        "Base": [base_returns.mean() * 12],
    })
    inputs_dict = {k: raw_params.get(k, "") for k in raw_params}
    raw_returns_dict = {"Base": pd.DataFrame(base_returns)}
    export_to_excel(inputs_dict, summary, raw_returns_dict, filename=args.output)


if __name__ == "__main__":
    main()
