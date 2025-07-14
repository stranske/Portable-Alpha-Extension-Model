from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    df_paths: pd.DataFrame | np.ndarray,
    liability: pd.Series | np.ndarray | None = None,
) -> go.Figure:
    """Return funding fan chart with median, confidence ribbon, optional liability."""
    arr = np.asarray(df_paths)
    median = np.median(arr, axis=0)
    conf = theme.THRESHOLDS.get("confidence", 0.95)
    lower = np.percentile(arr, 100 * (1 - conf), axis=0)
    upper = np.percentile(arr, 100 * conf, axis=0)

    months = np.arange(arr.shape[1])
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(
        go.Scatter(
            x=months, y=upper, mode="lines", line=dict(width=0), showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=lower,
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name=f"{int(conf * 100)}% CI",
        )
    )
    fig.add_trace(go.Scatter(x=months, y=median, mode="lines", name="Median"))
    if liability is not None:
        liab = np.asarray(liability).ravel()
        fig.add_trace(
            go.Scatter(
                x=months[: len(liab)],
                y=liab,
                mode="lines",
                line=dict(dash="dot"),
                name="Liability",
            )
        )
    fig.update_layout(xaxis_title="Month", yaxis_title="Return")
    return fig
