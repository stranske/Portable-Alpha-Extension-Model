from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    df_paths: pd.DataFrame | np.ndarray, *, window: int = 12, alpha: float = 0.95
) -> go.Figure:
    """Return rolling VaR line chart over the horizon."""
    arr = np.asarray(df_paths)
    n_months = arr.shape[1]
    var = np.full(n_months, np.nan)
    for t in range(window - 1, n_months):
        sub = arr[:, t - window + 1 : t + 1]
        cum = np.cumprod(1 + sub, axis=1)[:, -1] - 1
        var[t] = np.quantile(cum, 1 - alpha)
    months = np.arange(n_months)
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(go.Scatter(x=months, y=var, mode="lines", name="monthly_VaR"))
    fig.update_layout(xaxis_title="Month", yaxis_title="monthly_VaR")
    return fig
