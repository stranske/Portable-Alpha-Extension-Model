from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    df_paths: pd.DataFrame | np.ndarray,
    quantiles: Sequence[float] = (0.05, 0.95),
) -> go.Figure:
    """Return fan chart with custom quantile bounds."""
    arr = np.asarray(df_paths)
    q_low, q_high = quantiles[0], quantiles[-1]
    median = np.median(arr, axis=0)
    lower = np.percentile(arr, 100 * q_low, axis=0)
    upper = np.percentile(arr, 100 * q_high, axis=0)
    months = np.arange(arr.shape[1])
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(go.Scatter(x=months, y=upper, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=months,
            y=lower,
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name=f"{int(q_low*100)}–{int(q_high*100)}% CI",
        )
    )
    fig.add_trace(go.Scatter(x=months, y=median, mode="lines", name="Median"))
    fig.update_layout(xaxis_title="Month", yaxis_title="Return")
    return fig
