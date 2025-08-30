from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame, *, size_col: str = "Capital") -> go.Figure:
    """Return scatter of tracking error vs CVaR.

    Parameters
    ----------
    df_summary : pandas.DataFrame
        Must contain TrackingErr, CVaR and Agent columns.
    size_col : str, default "Capital"
        Optional column to scale marker size.
    """
    df = df_summary.copy()
    thr = theme.THRESHOLDS
    probs = df.get("ShortfallProb")
    probs = probs.fillna(theme.DEFAULT_SHORTFALL_PROB) if probs is not None else pd.Series(theme.DEFAULT_SHORTFALL_PROB, index=df.index)
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
    fig.add_trace(
        go.Scatter(
            x=df["TrackingErr"],
            y=df["CVaR"],
            mode="markers",
            marker=dict(size=10 + 20 * size / max_size, color=colors),
            text=df["Agent"],
            hovertemplate="%{text}<br>TE=%{x:.2%}<br>CVaR=%{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Tracking Error", yaxis_title="CVaR")
    return fig
