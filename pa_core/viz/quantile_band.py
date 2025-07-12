from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    df_paths: pd.DataFrame | np.ndarray, quantiles=(0.1, 0.9), window: int = 12
) -> go.Figure:
    """Return rolling quantile band over the median path."""
    arr = np.asarray(df_paths)
    cum = np.cumprod(1 + arr, axis=1)
    median = np.median(cum, axis=0)
    q_low, q_high = quantiles
    roll_low = pd.Series(median).rolling(window, min_periods=1).quantile(q_low)
    roll_high = pd.Series(median).rolling(window, min_periods=1).quantile(q_high)
    months = np.arange(arr.shape[1])
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(
        go.Scatter(x=months, y=roll_high, line=dict(width=0), showlegend=False)
    )
    fig.add_trace(
        go.Scatter(
            x=months, y=roll_low, fill="tonexty", line=dict(width=0), name="Band"
        )
    )
    fig.add_trace(go.Scatter(x=months, y=median, mode="lines", name="Median"))
    fig.update_layout(xaxis_title="Month", yaxis_title="Cumulative Return")
    return fig
