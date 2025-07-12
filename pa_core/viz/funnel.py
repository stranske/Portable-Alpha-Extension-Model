from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_paths: pd.DataFrame | np.ndarray, quantiles=(0.05, 0.95)) -> go.Figure:
    """Visualise widening distribution of cumulative returns."""
    arr = np.asarray(df_paths)
    cum = np.cumprod(1 + arr, axis=1)
    median = np.median(cum, axis=0)
    lower = np.quantile(cum, quantiles[0], axis=0)
    upper = np.quantile(cum, quantiles[1], axis=0)
    months = np.arange(arr.shape[1])
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(go.Scatter(x=months, y=upper, line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(x=months, y=lower, line=dict(width=0), fill="tonexty", name="Band")
    )
    fig.add_trace(go.Scatter(x=months, y=median, mode="lines", name="Median"))
    fig.update_layout(xaxis_title="Month", yaxis_title="Cumulative Return")
    return fig
