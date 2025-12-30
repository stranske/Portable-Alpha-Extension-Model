from __future__ import annotations

from typing import Mapping, cast

import pandas as pd
import plotly.graph_objects as go

from . import theme

__all__ = ["make", "series_from_sensitivity"]


def series_from_sensitivity(
    df: pd.DataFrame,
    *,
    param_col: str = "Parameter",
    value_col: str = "DeltaAbs",
) -> pd.Series:
    """Build a delta series for tornado charts from a sensitivity DataFrame."""
    if param_col not in df.columns or value_col not in df.columns:
        raise KeyError(f"Missing columns for tornado: {param_col}, {value_col}")
    series = df.set_index(param_col)[value_col].astype(float)
    series.attrs.update(df.attrs)
    return series


def make(
    contrib: Mapping[str, float] | pd.Series,
    title: str | None = None,
    *,
    metric: str | None = None,
    units: str | None = None,
    tickformat: str | None = None,
) -> go.Figure:
    """Render a tornado chart from a mapping/Series of contributions.

    Inputs may be a dict-like mapping of parameter -> delta contribution,
    or a pandas Series. Bars are sorted by absolute value descending.
    """
    if isinstance(contrib, pd.Series):
        series = cast(pd.Series, contrib)
    else:
        series = pd.Series({str(k): float(v) for k, v in contrib.items()})

    # Ensure numeric dtype for plotting
    series = series.astype(float)
    series_metric = metric or series.attrs.get("metric", "AnnReturn")
    series_units = units or series.attrs.get("units", "%")
    if tickformat is None:
        tickformat = series.attrs.get("tickformat")
    if tickformat is None and str(series_units).strip() in {"%", "pct", "percent"}:
        tickformat = ".2%"

    if series.empty:
        return go.Figure(layout_template=theme.TEMPLATE)

    series.index = series.index.astype(str)
    order = (
        pd.DataFrame(
            {
                "value": series,
                "abs": series.abs(),
                "name": series.index,
            }
        )
        .sort_values(["abs", "name"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    series = pd.Series(order["value"].to_numpy(), index=order["name"].to_list())
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(go.Bar(x=series.values, y=series.index.tolist(), orientation="h"))
    if series_metric and series_units:
        xaxis_title = f"Delta ({series_metric}, {series_units})"
    elif series_metric:
        xaxis_title = f"Delta ({series_metric})"
    else:
        xaxis_title = "Delta"
    fig.update_layout(
        title=title or "Sensitivity Tornado",
        xaxis_title=xaxis_title,
        yaxis_title="Parameter",
    )
    if tickformat:
        fig.update_xaxes(tickformat=str(tickformat))
    return fig
