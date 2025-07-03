from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(paths_map: Mapping[str, pd.DataFrame | np.ndarray]) -> go.Figure:
    """Return overlay of median cumulative return paths."""
    fig = go.Figure(layout_template=theme.TEMPLATE)
    for name, data in paths_map.items():
        arr = np.asarray(data)
        cum = np.cumprod(1 + arr, axis=1)
        median = np.median(cum, axis=0)
        months = np.arange(median.size)
        fig.add_trace(go.Scatter(x=months, y=median, mode="lines", name=name))
    fig.update_layout(xaxis_title="Month", yaxis_title="Cumulative Return")
    return fig
