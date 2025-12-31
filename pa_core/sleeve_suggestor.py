from __future__ import annotations

import itertools
from numbers import Real
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .config import ModelConfig
from .orchestrator import SimulatorOrchestrator


def _clamp_grid(grid: np.ndarray, min_value: float | None, max_value: float | None) -> np.ndarray:
    min_val = 0.0 if min_value is None else min_value
    max_val = float("inf") if max_value is None else max_value
    return grid[(grid >= min_val) & (grid <= max_val)]


def _pick_priority_combos(
    combos: Sequence[tuple[float, float]],
    ext_grid: Iterable[float],
    act_grid: Iterable[float],
    total: float,
    min_internal: float | None,
    max_internal: float | None,
) -> list[tuple[float, float]]:
    def internal_ok(value: float) -> bool:
        if min_internal is not None and value < min_internal:
            return False
        if max_internal is not None and value > max_internal:
            return False
        return True

    ext_vals = sorted(set(ext_grid))
    act_vals = sorted(set(act_grid))
    if not ext_vals or not act_vals:
        return []

    candidates = [
        (ext_vals[0], act_vals[0]),
        (ext_vals[0], act_vals[-1]),
        (ext_vals[-1], act_vals[0]),
        (ext_vals[-1], act_vals[-1]),
    ]

    target = total / 3
    ext_mid = min(ext_vals, key=lambda v: abs(v - target))
    act_mid = min(act_vals, key=lambda v: abs(v - target))
    candidates.append((ext_mid, act_mid))

    picked = []
    for ext_cap, act_cap in candidates:
        int_cap = total - ext_cap - act_cap
        if int_cap < 0:
            continue
        if not internal_ok(int_cap):
            continue
        if (ext_cap, act_cap) in combos and (ext_cap, act_cap) not in picked:
            picked.append((ext_cap, act_cap))
    return picked


def _coerce_metric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        try:
            metric = float(value)
        except ValueError:
            return None
    elif isinstance(value, (Real, np.floating, np.integer)):
        metric = float(value)
    else:
        return None
    if not np.isfinite(metric):
        return None
    return metric


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

    if step <= 0:
        raise ValueError("step must be positive")

    total = cfg.total_fund_capital
    step_size = total * step
    grid = np.arange(0.0, total + 1e-9, step_size)
    ext_grid = _clamp_grid(grid, min_external, max_external)
    act_grid = _clamp_grid(grid, min_active, max_active)

    if ext_grid.size == 0 or act_grid.size == 0:
        return pd.DataFrame()

    combos = [
        (ext_cap, act_cap)
        for ext_cap, act_cap in itertools.product(ext_grid, act_grid)
        if (total - ext_cap - act_cap) >= 0
    ]
    if max_evals is not None and len(combos) > max_evals:
        rng = np.random.default_rng(seed)
        priority = _pick_priority_combos(
            combos,
            ext_grid,
            act_grid,
            total,
            min_internal,
            max_internal,
        )
        if max_evals <= len(priority):
            combos = priority[:max_evals]
        else:
            remaining = [combo for combo in combos if combo not in priority]
            budget = max_evals - len(priority)
            if budget > 0 and remaining:
                idx = rng.choice(len(remaining), size=min(budget, len(remaining)), replace=False)
                sampled = [remaining[i] for i in idx]
            else:
                sampled = []
            combos = priority + sampled

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
        invalid_metrics = False
        for agent in ["ExternalPA", "ActiveExt", "InternalPA"]:
            sub = summary[summary["Agent"] == agent]
            if sub.empty:
                continue
            row = sub.iloc[0]
            te = _coerce_metric(row["TE"])
            bprob = _coerce_metric(row["BreachProb"])
            cvar = _coerce_metric(row["CVaR"])
            if te is None or bprob is None or cvar is None:
                invalid_metrics = True
                break
            metrics[f"{agent}_TE"] = float(te)
            metrics[f"{agent}_BreachProb"] = float(bprob)
            metrics[f"{agent}_CVaR"] = float(cvar)
            if constraint_scope in {"sleeves", "both"} and (
                te > max_te or bprob > max_breach or abs(cvar) > max_cvar
            ):
                meets = False
        if invalid_metrics:
            continue

        total_metrics: dict[str, float] = {}
        total_row = summary[summary["Agent"] == "Total"]
        if not total_row.empty:
            total_row = total_row.iloc[0]
            total_te = _coerce_metric(total_row["TE"])
            total_bprob = _coerce_metric(total_row["BreachProb"])
            total_cvar = _coerce_metric(total_row["CVaR"])
            if total_te is None or total_bprob is None or total_cvar is None:
                continue
            total_metrics = {
                "Total_TE": float(total_te),
                "Total_BreachProb": float(total_bprob),
                "Total_CVaR": float(total_cvar),
            }
            metrics.update(total_metrics)
            if constraint_scope in {"total", "both"} and (
                total_te > max_te or total_bprob > max_breach or abs(total_cvar) > max_cvar
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
