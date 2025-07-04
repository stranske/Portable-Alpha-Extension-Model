from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_paths: pd.DataFrame | np.ndarray) -> go.Figure:
    """Return cumulative return chart with a range slider."""
    arr = np.asarray(df_paths)
    months = np.arange(arr.shape[1])
    cum = np.cumprod(1 + arr, axis=1)
    median = np.median(cum, axis=0)

    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(go.Scatter(x=months, y=median, mode="lines", name="Median"))
    fig.update_layout(
        xaxis=dict(title="Month", rangeslider=dict(visible=True)),
        yaxis_title="Cumulative Return",
    )
    return fig
