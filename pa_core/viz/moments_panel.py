from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme


def make(df_paths: pd.DataFrame | np.ndarray, *, window: int = 12) -> go.Figure:
    """Return rolling skewness and kurtosis panel."""
    arr = np.asarray(df_paths)
    df = pd.DataFrame(arr)
    roll_skew = df.rolling(window, axis=1, min_periods=1).skew().mean(axis=0)
    roll_kurt = df.rolling(window, axis=1, min_periods=1).kurt().mean(axis=0)
    months = np.arange(arr.shape[1])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Skewness", "Kurtosis"))
    fig.add_trace(go.Scatter(x=months, y=roll_skew, name="Skew"), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=roll_kurt, name="Kurtosis"), row=2, col=1)
    fig.update_layout(template=theme.TEMPLATE, xaxis_title="Month", height=500)
    return fig
