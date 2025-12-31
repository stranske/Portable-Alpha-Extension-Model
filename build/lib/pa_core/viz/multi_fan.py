from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    df_paths: pd.DataFrame | np.ndarray,
    horizons: Sequence[int] = (12, 24, 36),
) -> go.Figure:
    """Return overlay of fan charts for multiple horizons."""
    arr = np.asarray(df_paths)
    fig = go.Figure(layout_template=theme.TEMPLATE)
    conf = theme.THRESHOLDS.get("confidence", 0.95)
    colors = ["blue", "orange", "green", "purple"]
    for idx, horizon in enumerate(horizons):
        h = min(int(horizon), arr.shape[1])
        sub = arr[:, :h]
        median = np.median(sub, axis=0)
        lower = np.percentile(sub, 100 * (1 - conf), axis=0)
        upper = np.percentile(sub, 100 * conf, axis=0)
        months = np.arange(h)
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(x=months, y=upper, mode="lines", line=dict(width=0), showlegend=False)
        )
        fig.add_trace(
            go.Scatter(
                x=months,
                y=lower,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                fillcolor=f"rgba(0,0,255,{0.2 + 0.1 * idx})",
                name=f"{h}m CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=months,
                y=median,
                mode="lines",
                name=f"{h}m Median",
                line=dict(color=color),
            )
        )
    fig.update_layout(xaxis_title="Month", yaxis_title="Return")
    return fig
