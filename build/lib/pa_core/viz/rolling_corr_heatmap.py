from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_paths: pd.DataFrame | np.ndarray, window: int = 12) -> go.Figure:
    """Return heatmap of month-on-month correlations with a rolling window.

    Parameters
    ----------
    df_paths:
        Array-like of shape ``(n_sim, n_months)``.
    window:
        Number of months over which to compute correlations.
    """
    arr = np.asarray(df_paths)
    n_months = arr.shape[1]
    max_lag = min(window, n_months - 1)
    z = np.full((max_lag, n_months), np.nan)

    for lag in range(1, max_lag + 1):
        for t in range(lag, n_months):
            x = arr[:, t]
            y = arr[:, t - lag]
            if x.size and y.size:
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = np.nan
            z[lag - 1, t] = corr

    y_labels: Sequence[str] = [f"lag {lag}" for lag in range(1, max_lag + 1)]
    fig = go.Figure(
        data=go.Heatmap(z=z, x=list(range(n_months)), y=y_labels),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Lag")
    return fig
