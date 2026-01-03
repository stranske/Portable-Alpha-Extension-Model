from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ..sleeve_suggestor import SLEEVE_AGENTS

__all__ = ["build_constraint_report"]


@dataclass(frozen=True)
class _ConstraintSpec:
    label: str
    metric: str
    limit: float


def _resolve_driver(
    summary: pd.DataFrame,
    metric: str,
    sleeves: Iterable[str],
) -> str:
    sleeve_rows = summary[summary["Agent"].isin(list(sleeves))]
    if sleeve_rows.empty:
        return ""
    metric_vals = pd.to_numeric(sleeve_rows[metric], errors="coerce")
    if metric_vals.isna().all():
        return ""
    idx = metric_vals.idxmax()
    if idx is None:
        return ""
    return str(sleeve_rows.loc[idx, "Agent"])


def build_constraint_report(
    summary: pd.DataFrame,
    *,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    sleeves: Iterable[str] = SLEEVE_AGENTS,
    total_agent: str = "Total",
) -> pd.DataFrame:
    """Return rows describing which constraint limits were breached."""
    if summary.empty or "Agent" not in summary.columns:
        return pd.DataFrame()

    specs = [
        _ConstraintSpec("Tracking error", "monthly_TE", max_te),
        _ConstraintSpec("Breach probability", "monthly_BreachProb", max_breach),
        _ConstraintSpec("Monthly CVaR", "monthly_CVaR", max_cvar),
    ]

    agents = summary["Agent"].astype(str)
    sleeve_set = {str(agent) for agent in sleeves}
    report_agents = {agent for agent in agents if agent in sleeve_set or agent == total_agent}
    if not report_agents:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for spec in specs:
        if spec.metric not in summary.columns:
            continue
        driver = _resolve_driver(summary, spec.metric, sleeve_set)
        for _, row in summary[summary["Agent"].isin(report_agents)].iterrows():
            value = pd.to_numeric(row.get(spec.metric), errors="coerce")
            if pd.isna(value):
                continue
            if float(value) <= float(spec.limit):
                continue
            agent = str(row["Agent"])
            rows.append(
                {
                    "Constraint": spec.label,
                    "Agent": agent,
                    "Metric": spec.metric,
                    "Limit": float(spec.limit),
                    "Value": float(value),
                    "Breach": float(value) - float(spec.limit),
                    "Driver": agent if agent in sleeve_set else driver,
                }
            )

    return pd.DataFrame(rows)
