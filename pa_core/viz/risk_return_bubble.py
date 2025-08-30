from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame, *, size_col: str = "Capital") -> go.Figure:
    """Return risk-return scatter with bubble sizing."""
    df = df_summary.copy()
    color = []
    thr = theme.THRESHOLDS
    probs = (
        df["ShortfallProb"] if "ShortfallProb" in df else pd.Series(0.0, index=df.index)
    )
    for prob in probs.fillna(0.0):
        if prob <= thr.get("shortfall_green", 0.05):
            color.append("green")
        elif prob <= thr.get("shortfall_amber", theme.LOW_BUFFER_THRESHOLD):
            color.append("orange")
        else:
            color.append("red")
    size = df[size_col] if size_col in df else pd.Series(1.0, index=df.index)
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(
        go.Scatter(
            x=df["AnnVol"],
            y=df["AnnReturn"],
            mode="markers",
            marker=dict(
                size=20 * size / float(size.max()), color=color, sizemode="diameter"
            ),
            text=df.get("Agent", ""),
            hovertemplate="%{text}<br>Vol=%{x:.2%}<br>Return=%{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Tracking Error", yaxis_title="Excess Return")
    return fig
