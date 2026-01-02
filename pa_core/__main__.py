from __future__ import annotations

import argparse
import json
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
        choices=["numpy"],
        help="Computation backend (numpy only; cupy/GPU acceleration is not available)",
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
                field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init
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
    from .random import spawn_agent_rngs_with_ids, spawn_rngs
    from .reporting import export_to_excel
    from .sim import draw_financing_series, draw_joint_returns
    from .sim.covariance import build_cov_matrix
    from .sim.metrics import summary_table
    from .sim.params import (
        build_covariance_return_overrides,
        build_params,
        resolve_covariance_inputs,
    )
    from .simulations import simulate_agents
    from .units import get_index_series_unit, normalize_index_series, normalize_return_inputs

    rng_returns = spawn_rngs(args.seed, 1)[0]
    fin_agent_names = ["internal", "external_pa", "active_ext"]
    fin_rngs, substream_ids = spawn_agent_rngs_with_ids(
        args.seed,
        fin_agent_names,
        legacy_order=args.legacy_agent_rng,
    )

    raw_params = cfg.model_dump()
    idx_series = load_index_returns(args.index)
    idx_series = normalize_index_series(idx_series, get_index_series_unit())
    mu_idx = float(idx_series.mean())
    vol_regime_value = getattr(cfg, "vol_regime", "single")
    if vol_regime_value not in ("single", "two_state"):
        raise ValueError(f"vol_regime must be 'single' or 'two_state', got {vol_regime_value!r}")
    vol_regime = cast(Literal["single", "two_state"], vol_regime_value)
    vol_regime_window = getattr(cfg, "vol_regime_window", 12)
    idx_sigma, _, _ = select_vol_regime_sigma(
        idx_series,
        regime=vol_regime,
        window=vol_regime_window,
    )
    n_samples = int(len(idx_series))

    return_inputs = normalize_return_inputs(cfg)
    sigma_H = return_inputs["sigma_H"]
    sigma_E = return_inputs["sigma_E"]
    sigma_M = return_inputs["sigma_M"]

    covariance_shrinkage_value = getattr(cfg, "covariance_shrinkage", "none")
    if covariance_shrinkage_value not in ("none", "ledoit_wolf"):
        raise ValueError(
            "covariance_shrinkage must be 'none' or 'ledoit_wolf', "
            f"got {covariance_shrinkage_value!r}"
        )
    covariance_shrinkage = cast(Literal["none", "ledoit_wolf"], covariance_shrinkage_value)
    cov = build_cov_matrix(
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

    sigma_vec, corr_mat = resolve_covariance_inputs(
        cov,
        idx_sigma=idx_sigma,
        sigma_h=sigma_H,
        sigma_e=sigma_E,
        sigma_m=sigma_M,
        rho_idx_H=cfg.rho_idx_H,
        rho_idx_E=cfg.rho_idx_E,
        rho_idx_M=cfg.rho_idx_M,
        rho_H_E=cfg.rho_H_E,
        rho_H_M=cfg.rho_H_M,
        rho_E_M=cfg.rho_E_M,
    )

    params = build_params(
        cfg,
        mu_idx=mu_idx,
        idx_sigma=float(sigma_vec[0]),
        return_overrides=build_covariance_return_overrides(sigma_vec, corr_mat),
    )

    N_SIMULATIONS = cfg.N_SIMULATIONS
    N_MONTHS = cfg.N_MONTHS

    r_beta, r_H, r_E, r_M = draw_joint_returns(
        n_months=N_MONTHS,
        n_sim=N_SIMULATIONS,
        params=params,
        rng=rng_returns,
    )
    corr_repair_info = params.get("_correlation_repair_info")
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
    if isinstance(corr_repair_info, dict) and corr_repair_info.get("repair_applied"):
        inputs_dict["correlation_repair_applied"] = True
        inputs_dict["correlation_repair_details"] = json.dumps(corr_repair_info)
    raw_returns_dict = {k: pd.DataFrame(v) for k, v in returns.items()}
    metadata = {"rng_seed": args.seed, "substream_ids": substream_ids}
    export_to_excel(inputs_dict, summary, raw_returns_dict, filename=args.output, metadata=metadata)
