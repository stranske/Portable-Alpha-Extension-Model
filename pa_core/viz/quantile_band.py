from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme

# Quantile band visualization constants
#
# These constants define the default quantile thresholds for visualization bands.
# Using named constants improves maintainability and documents the rationale
# for these specific threshold values in numerical calculations.

DEFAULT_LOWER_QUANTILE = 0.1
"""float: Lower quantile threshold for visualization bands.

This represents the 10th percentile, providing a 10% tolerance on the lower tail.
Used in combination with the upper quantile to create confidence intervals
in rolling quantile band visualizations.
"""

DEFAULT_UPPER_QUANTILE = 0.9
"""float: Upper quantile threshold for visualization bands.

This represents the 90th percentile, providing a 10% tolerance on the upper tail.
Combined with the 10th percentile lower quantile, this creates an 80% confidence
interval (excluding 20% of extreme values, 10% from each tail) for visualization
purposes in rolling quantile band plots.
"""


def make(
    df_paths: Union[pd.DataFrame, np.ndarray],
    quantiles: Tuple[float, float] = (DEFAULT_LOWER_QUANTILE, DEFAULT_UPPER_QUANTILE),
    window: int = 12,
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
    fig.add_trace(go.Scatter(x=months, y=roll_high, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=months, y=roll_low, fill="tonexty", line=dict(width=0), name="Band"))
    fig.add_trace(go.Scatter(x=months, y=median, mode="lines", name="Median"))
    fig.update_layout(xaxis_title="Month", yaxis_title="Cumulative Return")
    return fig
