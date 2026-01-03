from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pa_core.config import ModelConfig
from pa_core.facade import RunOptions, run_single, run_sweep


@dataclass(frozen=True)
class SettingCheck:
    name: str
    updates: dict[str, Any]
    kind: str = "summary"
    base_updates: dict[str, Any] | None = None
    agent: str = "Total"


def build_base_config(*, n_simulations: int, n_months: int) -> ModelConfig:
    data = {
        "Number of simulations": n_simulations,
        "Number of months": n_months,
        "financing_mode": "broadcast",
        "analysis_mode": "returns",
        "external_pa_capital": 100.0,
        "active_ext_capital": 50.0,
        "internal_pa_capital": 150.0,
        "total_fund_capital": 500.0,
    }
    return ModelConfig.model_validate(data)


def _build_index_series(cfg: ModelConfig, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.01, cfg.N_MONTHS))


def run_summary(cfg: ModelConfig, *, seed: int, agent: str = "Total") -> pd.Series:
    index_series = _build_index_series(cfg, seed)
    artifacts = run_single(cfg, index_series, RunOptions(seed=seed))
    summary = artifacts.summary
    if "Agent" not in summary.columns:
        raise ValueError("Summary output missing Agent column")
    target = agent
    if target in summary["Agent"].values:
        row_idx = summary.index[summary["Agent"] == target][0]
    else:
        row_idx = summary.index[0]
    numeric = summary.select_dtypes(include="number")
    return numeric.iloc[row_idx]


def run_inputs(cfg: ModelConfig, *, seed: int) -> dict[str, Any]:
    index_series = _build_index_series(cfg, seed)
    return run_single(cfg, index_series, RunOptions(seed=seed)).inputs


def summary_changed(base: pd.Series, changed: pd.Series, *, tol: float) -> bool:
    aligned_base, aligned_changed = base.align(changed, join="inner")
    na_mismatch = aligned_base.isna() ^ aligned_changed.isna()
    if bool(na_mismatch.any()):
        return True
    diff = (aligned_base - aligned_changed).abs().fillna(0.0)
    return bool((diff > tol).any())


def inputs_changed(
    base_inputs: dict[str, Any],
    changed_inputs: dict[str, Any],
    keys: Iterable[str],
) -> bool:
    for key in keys:
        if base_inputs.get(key) != changed_inputs.get(key):
            return True
    return False


def sweep_signature(summary: pd.DataFrame) -> tuple[int, int, int]:
    combos = summary["combination_id"].nunique() if "combination_id" in summary else 0
    return int(combos), int(summary.shape[0]), int(summary.shape[1])


def rebuild_config(cfg: ModelConfig, updates: dict[str, Any]) -> ModelConfig:
    data = cfg.model_dump()
    data.pop("agents", None)
    data.update(updates)
    return ModelConfig.model_validate(data)


def run_checks(
    checks: list[SettingCheck],
    *,
    seed: int,
    tol: float,
    n_simulations: int,
    n_months: int,
) -> list[str]:
    failures: list[str] = []
    for check in checks:
        base_cfg = build_base_config(n_simulations=n_simulations, n_months=n_months)
        if check.base_updates:
            base_cfg = rebuild_config(base_cfg, check.base_updates)
        changed_cfg = rebuild_config(base_cfg, check.updates)

        if check.kind == "summary":
            base_metrics = run_summary(base_cfg, seed=seed, agent=check.agent)
            changed_metrics = run_summary(changed_cfg, seed=seed, agent=check.agent)
            changed = summary_changed(base_metrics, changed_metrics, tol=tol)
        elif check.kind == "inputs":
            base_inputs = run_single(
                base_cfg, _build_index_series(base_cfg, seed), RunOptions(seed=seed)
            ).inputs
            changed_inputs = run_single(
                changed_cfg, _build_index_series(changed_cfg, seed), RunOptions(seed=seed)
            ).inputs
            changed = inputs_changed(base_inputs, changed_inputs, check.updates.keys())
        elif check.kind == "sweep":
            base_summary = run_sweep(
                base_cfg, _build_index_series(base_cfg, seed), None, RunOptions(seed=seed)
            ).summary
            changed_summary = run_sweep(
                changed_cfg, _build_index_series(changed_cfg, seed), None, RunOptions(seed=seed)
            ).summary
            changed = sweep_signature(base_summary) != sweep_signature(changed_summary)
        else:
            raise ValueError(f"Unknown check kind: {check.kind}")

        if not changed:
            failures.append(check.name)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate wizard settings affect outputs")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility")
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=200,
        help="Number of simulations for each check",
    )
    parser.add_argument(
        "--n-months",
        type=int,
        default=12,
        help="Number of months in each simulation run",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Numeric tolerance for output change checks",
    )
    args = parser.parse_args(argv)

    schedule_template = Path("config/margin_schedule_template.csv")
    if not schedule_template.exists():
        raise FileNotFoundError(schedule_template)

    with tempfile.TemporaryDirectory() as tmp_dir:
        alt_schedule = Path(tmp_dir) / "alt_schedule.csv"
        alt_schedule.write_text("term,multiplier\n1,1.5\n12,2.5\n", encoding="utf-8")

        checks = [
            SettingCheck("N_SIMULATIONS", {"N_SIMULATIONS": args.n_simulations + 100}),
            SettingCheck("N_MONTHS", {"N_MONTHS": args.n_months + 6}),
            SettingCheck(
                "analysis_mode",
                {
                    "analysis_mode": "capital",
                    "max_external_combined_pct": 10.0,
                    "external_step_size_pct": 10.0,
                },
                kind="sweep",
                base_updates={
                    "analysis_mode": "returns",
                    "in_house_return_min_pct": 2.0,
                    "in_house_return_max_pct": 4.0,
                    "in_house_return_step_pct": 2.0,
                    "in_house_vol_min_pct": 1.0,
                    "in_house_vol_max_pct": 1.0,
                    "in_house_vol_step_pct": 1.0,
                    "alpha_ext_return_min_pct": 1.0,
                    "alpha_ext_return_max_pct": 3.0,
                    "alpha_ext_return_step_pct": 2.0,
                    "alpha_ext_vol_min_pct": 2.0,
                    "alpha_ext_vol_max_pct": 2.0,
                    "alpha_ext_vol_step_pct": 1.0,
                },
            ),
            SettingCheck(
                "financing_mode",
                {"financing_mode": "per_path"},
                base_updates={
                    "internal_financing_sigma_month": 0.02,
                    "ext_pa_financing_sigma_month": 0.02,
                    "act_ext_financing_sigma_month": 0.02,
                },
            ),
            SettingCheck("total_fund_capital", {"total_fund_capital": 600.0}),
            SettingCheck("external_pa_capital", {"external_pa_capital": 130.0}),
            SettingCheck("active_ext_capital", {"active_ext_capital": 80.0}),
            SettingCheck("internal_pa_capital", {"internal_pa_capital": 200.0}),
            SettingCheck(
                "w_beta_H",
                {"w_beta_H": 0.6, "w_alpha_H": 0.4},
                agent="Base",
            ),
            SettingCheck(
                "w_alpha_H",
                {"w_beta_H": 0.4, "w_alpha_H": 0.6},
                agent="Base",
            ),
            SettingCheck("theta_extpa", {"theta_extpa": 0.7}),
            SettingCheck("active_share", {"active_share": 0.7}),
            SettingCheck("mu_H", {"mu_H": 0.06}),
            SettingCheck("mu_E", {"mu_E": 0.07}),
            SettingCheck("mu_M", {"mu_M": 0.05}),
            SettingCheck("sigma_H", {"sigma_H": 0.02}),
            SettingCheck("sigma_E", {"sigma_E": 0.03}),
            SettingCheck("sigma_M", {"sigma_M": 0.03}),
            SettingCheck("rho_idx_H", {"rho_idx_H": 0.15}),
            SettingCheck("rho_idx_E", {"rho_idx_E": 0.1}),
            SettingCheck("rho_idx_M", {"rho_idx_M": 0.1}),
            SettingCheck("rho_H_E", {"rho_H_E": 0.2}),
            SettingCheck("rho_H_M", {"rho_H_M": 0.2}),
            SettingCheck("rho_E_M", {"rho_E_M": 0.1}),
            SettingCheck(
                "risk_metrics",
                {
                    "risk_metrics": [
                        "Return",
                        "Risk",
                        "terminal_ShortfallProb",
                        "monthly_BreachProb",
                    ]
                },
                kind="inputs",
            ),
            SettingCheck("reference_sigma", {"reference_sigma": 0.015}),
            SettingCheck("volatility_multiple", {"volatility_multiple": 4.0}),
            SettingCheck(
                "financing_model",
                {"financing_model": "schedule", "financing_schedule_path": schedule_template},
            ),
            SettingCheck(
                "financing_schedule_path",
                {"financing_schedule_path": alt_schedule},
                base_updates={
                    "financing_model": "schedule",
                    "financing_schedule_path": schedule_template,
                },
            ),
            SettingCheck(
                "financing_term_months",
                {"financing_term_months": 6.0},
                base_updates={
                    "financing_model": "schedule",
                    "financing_schedule_path": schedule_template,
                },
            ),
        ]

        failures = run_checks(
            checks,
            seed=args.seed,
            tol=args.tolerance,
            n_simulations=args.n_simulations,
            n_months=args.n_months,
        )

    if failures:
        print("Failed settings:", ", ".join(sorted(failures)))
        return 1

    print("All settings produced output changes.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
