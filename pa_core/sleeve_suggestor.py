from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd

from .config import ModelConfig
from .orchestrator import SimulatorOrchestrator
from .sim.metrics import summary_table


def suggest_sleeve_sizes(
    cfg: ModelConfig,
    idx_series: pd.Series,
    *,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    step: float = 0.25,
    min_external: float | None = None,
    max_external: float | None = None,
    min_active: float | None = None,
    max_active: float | None = None,
    min_internal: float | None = None,
    max_internal: float | None = None,
    sort_by: str = "risk_score",
    seed: int | None = None,
    max_evals: int | None = 500,
    constraint_scope: Literal["sleeves", "total", "both"] = "sleeves",
) -> pd.DataFrame:
    """Suggest sleeve allocations that respect risk constraints.

    Performs a simple grid search over capital allocations for the three
    sleeves (external portable alpha, active extension and internal PA).
    Combinations where any sleeve breaches the supplied risk limits are
    discarded.

    Parameters
    ----------
    cfg:
        Base :class:`~pa_core.config.ModelConfig` used as a template.
    idx_series:
        Benchmark return series used by :class:`SimulatorOrchestrator`.
    max_te:
        Maximum allowed tracking error per sleeve.
    max_breach:
        Maximum allowed breach probability per sleeve.
    max_cvar:
        Absolute CVaR cap per sleeve.
    step:
        Grid step as a fraction of ``total_fund_capital``.
    seed:
        Optional random seed for reproducibility.
    max_evals:
        If set and the Cartesian grid would exceed this number of
        combinations, a random subset of at most ``max_evals`` points is
        evaluated. This prevents exponential runtime as ``step`` becomes
        small.
    constraint_scope:
        Apply constraints to per-sleeve metrics, total portfolio metrics,
        or both.

    Returns
    -------
    pandas.DataFrame
        Table of feasible capital combinations and associated metrics.
    """

    if constraint_scope not in {"sleeves", "total", "both"}:
        raise ValueError("constraint_scope must be one of: sleeves, total, both")

    total = cfg.total_fund_capital
    grid = np.arange(0.0, total + 1e-9, total * step)

    combos = [
        (ext_cap, act_cap)
        for ext_cap, act_cap in itertools.product(grid, repeat=2)
        if (total - ext_cap - act_cap) >= 0
    ]
    if max_evals is not None and len(combos) > max_evals:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combos), size=max_evals, replace=False)
        combos = [combos[i] for i in idx]

    records: list[dict[str, float]] = []
    for ext_cap, act_cap in combos:
        int_cap = total - ext_cap - act_cap

        # Bounds filtering (if provided)
        if min_external is not None and ext_cap < min_external:
            continue
        if max_external is not None and ext_cap > max_external:
            continue
        if min_active is not None and act_cap < min_active:
            continue
        if max_active is not None and act_cap > max_active:
            continue
        if min_internal is not None and int_cap < min_internal:
            continue
        if max_internal is not None and int_cap > max_internal:
            continue

        test_cfg = cfg.model_copy(
            update={
                "external_pa_capital": float(ext_cap),
                "active_ext_capital": float(act_cap),
                "internal_pa_capital": float(int_cap),
            }
        )
        orch = SimulatorOrchestrator(test_cfg, idx_series)
        returns, summary = orch.run(seed=seed)

        meets = True
        metrics: dict[str, float] = {}
        for agent in ["ExternalPA", "ActiveExt", "InternalPA"]:
            sub = summary[summary["Agent"] == agent]
            if sub.empty:
                continue
            row = sub.iloc[0]
            te = row["TE"] if row["TE"] is not None else 0.0
            bprob = row["BreachProb"]
            cvar = row["CVaR"]
            metrics[f"{agent}_TE"] = float(te)
            metrics[f"{agent}_BreachProb"] = float(bprob)
            metrics[f"{agent}_CVaR"] = float(cvar)
            if constraint_scope in {"sleeves", "both"} and (
                te > max_te or bprob > max_breach or abs(cvar) > max_cvar
            ):
                meets = False

        total_metrics: dict[str, float] = {}
        if "Base" in returns:
            total_returns = np.zeros_like(returns["Base"])
            weights = {
                "ExternalPA": ext_cap / total if total else 0.0,
                "ActiveExt": act_cap / total if total else 0.0,
                "InternalPA": int_cap / total if total else 0.0,
            }
            for name, weight in weights.items():
                if weight and name in returns:
                    total_returns += weight * returns[name]

            total_summary = summary_table(
                {"Base": returns["Base"], "Total": total_returns}, benchmark="Base"
            )
            total_row = total_summary[total_summary["Agent"] == "Total"].iloc[0]
            total_te = total_row["TE"] if total_row["TE"] is not None else 0.0
            total_bprob = total_row["BreachProb"]
            total_cvar = total_row["CVaR"]
            total_metrics = {
                "Total_TE": float(total_te),
                "Total_BreachProb": float(total_bprob),
                "Total_CVaR": float(total_cvar),
            }
            metrics.update(total_metrics)
            if constraint_scope in {"total", "both"} and (
                total_te > max_te
                or total_bprob > max_breach
                or abs(total_cvar) > max_cvar
            ):
                meets = False
        if meets:
            record = {
                "external_pa_capital": float(ext_cap),
                "active_ext_capital": float(act_cap),
                "internal_pa_capital": float(int_cap),
            }
            record.update(metrics)
            # Composite risk score (lower is better): TE + BreachProb + |CVaR|
            score = 0.0
            if constraint_scope in {"sleeves", "both"}:
                for ag in ["ExternalPA", "ActiveExt", "InternalPA"]:
                    score += record.get(f"{ag}_TE", 0.0)
                    score += record.get(f"{ag}_BreachProb", 0.0)
                    score += abs(record.get(f"{ag}_CVaR", 0.0))
            if constraint_scope in {"total", "both"}:
                score += record.get("Total_TE", 0.0)
                score += record.get("Total_BreachProb", 0.0)
                score += abs(record.get("Total_CVaR", 0.0))
            record["risk_score"] = score
            records.append(record)
    df = pd.DataFrame.from_records(records)
    if not df.empty and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True).reset_index(drop=True)
    return df
