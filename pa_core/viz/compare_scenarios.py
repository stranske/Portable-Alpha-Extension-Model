from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd
import plotly.graph_objects as go

from . import risk_return, theme

_RETURN_LABELS = {
    "terminal_ExcessReturn": "Annualized Excess Return",
    "terminal_AnnReturn": "Annualized Return",
}
_CVAR_LABELS = {
    "monthly_CVaR": "Monthly CVaR",
    "terminal_CVaR": "Terminal CVaR",
}


def compare_scenarios(results_list: Iterable[Any]) -> dict[str, go.Figure]:
    """Return comparison plots for a list of scenario results."""
    scenarios = list(results_list)
    if not scenarios:
        raise ValueError("results_list must contain at least one scenario")

    rows = []
    for idx, item in enumerate(scenarios):
        summary = _extract_summary(item)
        label = _extract_label(item, idx)
        row = _select_summary_row(summary)
        row_dict = row.to_dict()
        row_dict["Agent"] = label
        rows.append(row_dict)

    compare_df = pd.DataFrame(rows)
    risk_fig = risk_return.make(compare_df)

    cvar_col = _pick_column(compare_df, _CVAR_LABELS, "CVaR")
    return_col = _pick_column(compare_df, _RETURN_LABELS, "Return")
    cvar_fig = go.Figure(layout_template=theme.TEMPLATE)
    cvar_fig.add_trace(
        go.Scatter(
            x=compare_df[cvar_col],
            y=compare_df[return_col],
            mode="markers",
            marker=dict(size=12),
            text=compare_df["Agent"],
            hovertemplate=(
                f"%{{text}}<br>{_CVAR_LABELS.get(cvar_col, cvar_col)}=%{{x:.2%}}"
                f"<br>{_RETURN_LABELS.get(return_col, return_col)}=%{{y:.2%}}<extra></extra>"
            ),
        )
    )
    cvar_fig.update_layout(
        xaxis_title=_CVAR_LABELS.get(cvar_col, cvar_col),
        yaxis_title=_RETURN_LABELS.get(return_col, return_col),
        template=theme.TEMPLATE,
    )

    return {"risk_return": risk_fig, "cvar_return": cvar_fig}


def _extract_summary(item: Any) -> pd.DataFrame:
    if isinstance(item, pd.DataFrame):
        return item
    if isinstance(item, Mapping) and isinstance(item.get("summary"), pd.DataFrame):
        return item["summary"]
    summary = getattr(item, "summary", None)
    if isinstance(summary, pd.DataFrame):
        return summary
    raise TypeError("Each scenario must provide a pandas DataFrame summary")


def _extract_label(item: Any, idx: int) -> str:
    if isinstance(item, Mapping):
        for key in ("label", "name", "scenario", "scenario_id"):
            if key in item and item[key] is not None:
                return str(item[key])
        if "combination_id" in item and item["combination_id"] is not None:
            return f"Scenario {item['combination_id']}"
    label = getattr(item, "label", None)
    if label is not None:
        return str(label)
    return f"Scenario {idx + 1}"


def _select_summary_row(summary: pd.DataFrame) -> pd.Series:
    if "Agent" not in summary.columns or summary.empty:
        return summary.iloc[0]
    for agent in ("Total", "Base"):
        match = summary[summary["Agent"] == agent]
        if not match.empty:
            return match.iloc[0]
    return summary.iloc[0]


def _pick_column(df: pd.DataFrame, labels: Mapping[str, str], name: str) -> str:
    for col in labels:
        if col in df.columns and df[col].notna().any():
            return col
    raise KeyError(f"No {name} column found in scenario comparison data")
