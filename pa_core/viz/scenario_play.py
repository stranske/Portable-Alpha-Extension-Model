from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_paths: pd.DataFrame | np.ndarray, *, interval: int = 500) -> go.Figure:
    """Add play buttons to step through cumulative returns."""
    arr = np.asarray(df_paths)
    cum = np.cumprod(1 + arr, axis=1)
    median = np.median(cum, axis=0)
    months = np.arange(arr.shape[1])
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(go.Scatter(x=[0], y=[median[0]], mode="lines"))
    frames = [go.Frame(data=[go.Scatter(x=months[: i + 1], y=median[: i + 1])]) for i in range(len(months))]
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": interval}}]}
                ],
            }
        ]
    )
    return fig
