from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme


def _rolling_drawdown(paths: np.ndarray, window: int) -> np.ndarray:
    # Compute rolling drawdown on the median cumulative return path
    cum = np.cumprod(1 + paths, axis=1)
    median = np.median(cum, axis=0)
    roll_max = pd.Series(median).cummax()
    dd = 1 - median / roll_max
    return pd.Series(dd).rolling(window, min_periods=1).max().to_numpy()  # type: ignore[no-any-return]


def _rolling_te(paths: np.ndarray, window: int) -> np.ndarray:
    # Rolling tracking error (std deviation across sims)
    ser = pd.Series(np.std(paths, axis=0))
    rolled = ser.rolling(window, min_periods=1).mean()
    return np.asarray(rolled) * np.sqrt(12)  # type: ignore[no-any-return]


def _rolling_sharpe(paths: np.ndarray, window: int) -> np.ndarray:
    returns = pd.DataFrame(paths)
    roll_mean = returns.rolling(window, axis=1, min_periods=1).mean()
    roll_std = returns.rolling(window, axis=1, min_periods=1).std()
    sharpe = roll_mean / roll_std
    return sharpe.mean(axis=0).to_numpy() * np.sqrt(12)  # type: ignore[no-any-return]


def make(df_paths: pd.DataFrame | np.ndarray, window: int = 12) -> go.Figure:
    """Return rolling metrics panel with drawdown, TE and Sharpe."""
    arr = np.asarray(df_paths)
    dd = _rolling_drawdown(arr, window)
    te = _rolling_te(arr, window)
    sr = _rolling_sharpe(arr, window)
    months = np.arange(arr.shape[1])
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Drawdown", "Tracking Error", "Sharpe"),
    )
    fig.add_trace(go.Scatter(x=months, y=dd, name="Drawdown"), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=te, name="TE"), row=2, col=1)
    fig.add_trace(go.Scatter(x=months, y=sr, name="Sharpe"), row=3, col=1)
    fig.update_layout(template=theme.TEMPLATE, xaxis_title="Month", height=600)
    return fig
