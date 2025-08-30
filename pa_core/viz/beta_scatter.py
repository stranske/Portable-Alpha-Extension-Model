from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    df_summary: pd.DataFrame,
    *,
    te_col: str = "TrackingErr",
    beta_col: str = "Beta",
    size_col: str = "Capital",
) -> go.Figure:
    """Return scatter of tracking error vs beta exposure."""
    df = df_summary.copy()
    thr = theme.THRESHOLDS
    probs = df.get("ShortfallProb")
    probs = probs.fillna(0.0) if probs is not None else pd.Series(0.0, index=df.index)
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
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(
        go.Scatter(
            x=df[te_col],
            y=df[beta_col],
            mode="markers",
            marker=dict(
                size=20 * size / float(size.max()), color=colors, sizemode="diameter"
            ),
            text=df.get("Agent", ""),
            hovertemplate="%{text}<br>TE=%{x:.2%}<br>Beta=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Tracking Error", yaxis_title="Beta Exposure")
    return fig
