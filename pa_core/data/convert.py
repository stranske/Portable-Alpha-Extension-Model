from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from .loaders import load_parameters
from ..config import load_config

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
    "risk_metrics": "risk_metrics",
}


def convert(csv_path: str | Path, yaml_path: str | Path) -> None:
    """Convert legacy parameters CSV to YAML configuration."""

    raw = load_parameters(csv_path, LABEL_MAP)
    cfg = load_config(raw)
    Path(yaml_path).write_text(yaml.safe_dump(cfg.model_dump()))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert parameters CSV to YAML")
    parser.add_argument("csv", help="Input parameters CSV file")
    parser.add_argument(
        "yaml", nargs="?", help="Output YAML file (default: params.yml)"
    )
    args = parser.parse_args(argv)
    out = args.yaml or Path(args.csv).with_suffix(".yml")
    convert(args.csv, out)


if __name__ == "__main__":  # pragma: no cover
    main()
