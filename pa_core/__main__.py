from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from typing import Literal, Optional, Sequence, cast

import pandas as pd

from .backend import resolve_and_set_backend
from .config import load_config
from .validators import select_vol_regime_sigma


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Portable Alpha simulation")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--index", required=True, help="Index returns CSV")
    parser.add_argument("--output", default="Outputs.xlsx", help="Output workbook")
    parser.add_argument(
        "--backend",
        choices=["numpy", "cupy"],
        help="Computation backend",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulations",
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
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    return_overrides: dict[str, float | str] = {}
    if args.return_distribution is not None:
        return_overrides["return_distribution"] = args.return_distribution
    if args.return_t_df is not None:
        return_overrides["return_t_df"] = args.return_t_df
    if args.return_copula is not None:
        return_overrides["return_copula"] = args.return_copula
    if return_overrides:
        if is_dataclass(cfg):
            base_data = {
                field.name: getattr(cfg, field.name)
                for field in fields(cfg)
                if field.init
            }
        else:
            base_data = cfg.model_dump()
        cfg = cfg.__class__.model_validate({**base_data, **return_overrides})
    backend_choice = resolve_and_set_backend(args.backend, cfg)
    args.backend = backend_choice
    print(f"[BACKEND] Using backend: {backend_choice}")

    # Import backend-dependent modules after setting the backend.
    from .agents.registry import build_from_config
    from .data import load_index_returns
    from .random import spawn_agent_rngs, spawn_rngs
    from .reporting import export_to_excel
    from .sim import draw_financing_series, draw_joint_returns
    from .sim.covariance import build_cov_matrix
    from .sim.metrics import summary_table
    from .simulations import simulate_agents

    rng_returns = spawn_rngs(args.seed, 1)[0]
    fin_rngs = spawn_agent_rngs(
        args.seed,
        ["internal", "external_pa", "active_ext"],
    )

    raw_params = cfg.model_dump()
    idx_series = load_index_returns(args.index)
    mu_idx = float(idx_series.mean())
    vol_regime_value = getattr(cfg, "vol_regime", "single")
    if vol_regime_value not in ("single", "two_state"):
        raise ValueError(
            f"vol_regime must be 'single' or 'two_state', got {vol_regime_value!r}"
        )
    vol_regime = cast(Literal["single", "two_state"], vol_regime_value)
    vol_regime_window = getattr(cfg, "vol_regime_window", 12)
    idx_sigma, _, _ = select_vol_regime_sigma(
        idx_series,
        regime=vol_regime,
        window=vol_regime_window,
    )
    n_samples = int(len(idx_series))

    mu_H = cfg.mu_H
    sigma_H = cfg.sigma_H
    mu_E = cfg.mu_E
    sigma_E = cfg.sigma_E
    mu_M = cfg.mu_M
    sigma_M = cfg.sigma_M

    covariance_shrinkage_value = getattr(cfg, "covariance_shrinkage", "none")
    if covariance_shrinkage_value not in ("none", "ledoit_wolf"):
        raise ValueError(
            "covariance_shrinkage must be 'none' or 'ledoit_wolf', "
            f"got {covariance_shrinkage_value!r}"
        )
    covariance_shrinkage = cast(
        Literal["none", "ledoit_wolf"], covariance_shrinkage_value
    )
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
        covariance_shrinkage=covariance_shrinkage,
        n_samples=n_samples,
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
        "return_distribution": cfg.return_distribution,
        "return_t_df": cfg.return_t_df,
        "return_copula": cfg.return_copula,
        "return_distribution_idx": cfg.return_distribution_idx,
        "return_distribution_H": cfg.return_distribution_H,
        "return_distribution_E": cfg.return_distribution_E,
        "return_distribution_M": cfg.return_distribution_M,
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
