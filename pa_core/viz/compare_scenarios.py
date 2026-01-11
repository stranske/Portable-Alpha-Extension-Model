from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..sim import metrics
from . import risk_return, theme, utils

_RETURN_LABELS = {
    "terminal_ExcessReturn": "Annualized Excess Return",
    "terminal_AnnReturn": "Annualized Return",
}
_CVAR_LABELS = {
    "monthly_CVaR": "Monthly CVaR",
    "terminal_CVaR": "Terminal CVaR",
}
_RISK_METRICS = {
    "monthly_VaR": "Monthly VaR",
    "monthly_CVaR": "Monthly CVaR",
    "monthly_MaxDD": "Max Drawdown",
}


def compare_scenarios(results_list: Iterable[Any]) -> dict[str, go.Figure]:
    """Return comparison plots for a list of scenario results."""
    scenarios = list(results_list)
    if not scenarios:
        raise ValueError("results_list must contain at least one scenario")

    rows = []
    dist_rows = []
    for idx, item in enumerate(scenarios):
        summary = _extract_summary(item)
        label = _extract_label(item, idx)
        returns = _extract_returns(item)
        row = _select_summary_row(summary)
        row_dict = row.to_dict()
        row_dict["Agent"] = label
        rows.append(row_dict)
        dist_rows.append((label, returns))

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

    return_dist_fig = _make_return_distribution(dist_rows)
    risk_metrics_fig = _make_risk_metric_bars(compare_df)

    return {
        "risk_return": risk_fig,
        "cvar_return": cvar_fig,
        "return_distribution": return_dist_fig,
        "risk_metrics": risk_metrics_fig,
    }


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


def _extract_returns(item: Any) -> np.ndarray:
    raw_returns = None
    returns = None
    if isinstance(item, Mapping):
        raw_returns = item.get("raw_returns")
        returns = item.get("returns")
    else:
        raw_returns = getattr(item, "raw_returns", None)
        returns = getattr(item, "returns", None)

    if raw_returns is not None:
        return _coerce_returns(raw_returns)
    if returns is not None:
        return _coerce_returns(returns)
    raise TypeError("Each scenario must provide returns or raw_returns for distributions")


def _coerce_returns(data: Any) -> np.ndarray:
    values = data
    if isinstance(data, Mapping):
        key = _select_returns_key(data)
        values = data[key]
    if isinstance(values, pd.DataFrame):
        arr = utils.safe_to_numpy(values)
    else:
        arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _select_returns_key(data: Mapping[str, Any]) -> str:
    for key in ("Total", "Base"):
        if key in data:
            return key
    for key, value in data.items():
        if value is not None:
            return key
    raise KeyError("No return series found in scenario data")


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


def _make_return_distribution(dist_rows: list[tuple[str, np.ndarray]]) -> go.Figure:
    terminal_sets = []
    for label, arr in dist_rows:
        terminal = _terminal_compounded_returns(arr)
        terminal_sets.append((label, terminal))

    mins = [np.min(vals) for _, vals in terminal_sets]
    maxs = [np.max(vals) for _, vals in terminal_sets]
    lower = float(min(mins))
    upper = float(max(maxs))
    if lower == upper:
        lower -= 0.01
        upper += 0.01

    grid = np.linspace(lower, upper, 200)
    fig = go.Figure(layout_template=theme.TEMPLATE)
    for label, terminal in terminal_sets:
        hist = go.Histogram(
            x=terminal,
            nbinsx=40,
            histnorm="probability density",
            name=label,
            opacity=0.45,
            legendgroup=label,
        )
        density = _kde_density(terminal, grid)
        line = go.Scatter(
            x=grid,
            y=density,
            mode="lines",
            name=f"{label} density",
            legendgroup=label,
            showlegend=False,
        )
        fig.add_trace(hist)
        fig.add_trace(line)

    fig.update_layout(
        xaxis_title="Terminal Compounded Return",
        yaxis_title="Density",
        barmode="overlay",
    )
    return fig


def _make_risk_metric_bars(compare_df: pd.DataFrame) -> go.Figure:
    missing = [col for col in _RISK_METRICS if col not in compare_df.columns]
    if missing:
        raise KeyError(f"Missing risk metric columns: {', '.join(missing)}")
    fig = go.Figure(layout_template=theme.TEMPLATE)
    metric_labels = [_RISK_METRICS[col] for col in _RISK_METRICS]
    for _, row in compare_df.iterrows():
        fig.add_trace(
            go.Bar(
                name=row["Agent"],
                x=metric_labels,
                y=[row[col] for col in _RISK_METRICS],
                hovertemplate="%{fullData.name}<br>%{x}=%{y:.2%}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        xaxis_title="Risk Metric",
        yaxis_title="Value",
        yaxis_tickformat=".2%",
    )
    return fig


def _terminal_compounded_returns(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.size == 0:
        raise ValueError("Return series is empty")
    terminal = metrics.compound(data)[:, -1]
    terminal = terminal[np.isfinite(terminal)]
    if terminal.size == 0:
        raise ValueError("Return series has no finite values")
    return terminal


def _kde_density(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    data = _downsample(samples)
    if data.size < 2:
        return np.zeros_like(grid)
    std = np.std(data, ddof=1)
    if std == 0:
        return np.zeros_like(grid)
    bandwidth = 1.06 * std * data.size ** (-0.2)
    if bandwidth <= 0:
        return np.zeros_like(grid)
    diffs = (grid[:, None] - data[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs**2)
    return np.mean(density, axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def _downsample(samples: np.ndarray, max_samples: int = 5000) -> np.ndarray:
    if samples.size <= max_samples:
        return samples
    idx = np.linspace(0, samples.size - 1, max_samples).astype(int)
    return samples[idx]
