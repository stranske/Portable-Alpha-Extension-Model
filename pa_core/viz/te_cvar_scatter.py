from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame, *, size_col: str = "Capital") -> go.Figure:
    """Return scatter of monthly tracking error vs monthly CVaR.

    Parameters
    ----------
    df_summary : pandas.DataFrame
        Must contain monthly_TE (or TrackingErr), monthly_CVaR, and Agent columns.
    size_col : str, default "Capital"
        Optional column to scale marker size.
    """
    df = df_summary.copy()
    thr = theme.THRESHOLDS
    probs = df.get("terminal_ShortfallProb")
    probs = (
        probs.fillna(theme.DEFAULT_SHORTFALL_PROB)
        if probs is not None
        else pd.Series(theme.DEFAULT_SHORTFALL_PROB, index=df.index)
    )
    colors = []
    for p in probs:
        if p <= thr.get("shortfall_green", 0.05):
            colors.append("green")
        elif p <= thr.get("shortfall_amber", theme.LOW_BUFFER_THRESHOLD):
            colors.append("orange")
        else:
            colors.append("red")
    size = df.get(size_col)
    size = size if size is not None else pd.Series(1.0, index=df.index)
    max_size = size.max() if size.max() > 0 else 1.0
    fig = go.Figure(layout_template=theme.TEMPLATE)
    x_col = "monthly_TE" if "monthly_TE" in df else "TrackingErr"
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df["monthly_CVaR"],
            mode="markers",
            marker=dict(size=10 + 20 * size / max_size, color=colors),
            text=df["Agent"],
            hovertemplate="%{text}<br>monthly_TE=%{x:.2%}<br>monthly_CVaR=%{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="monthly_TE", yaxis_title="monthly_CVaR")
    return fig
