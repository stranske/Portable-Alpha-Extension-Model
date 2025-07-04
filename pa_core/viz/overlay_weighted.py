from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(paths_map: Mapping[str, Tuple[pd.DataFrame | np.ndarray, float]]) -> go.Figure:
    """Return overlay of median cumulative return paths weighted by capital."""
    first = next(iter(paths_map.values()))[0]
    months = np.arange(np.asarray(first).shape[1])
    fig = go.Figure(layout_template=theme.TEMPLATE)
    weights = {name: weight for name, (_, weight) in paths_map.items()}
    max_w = max(weights.values()) if weights else 1.0
    # Individual paths with line width based on weight
    for name, (data, weight) in paths_map.items():
        arr = np.asarray(data)
        cum = np.cumprod(1 + arr, axis=1)
        median = np.median(cum, axis=0)
        fig.add_trace(
            go.Scatter(
                x=months,
                y=median,
                mode="lines",
                name=name,
                line=dict(width=2 + 4 * weight / max_w),
            )
        )
    # Combined weighted path
    tot_w = sum(weights.values())
    if tot_w > 0:
        composite = np.zeros_like(months, dtype=float)
        for name, (data, weight) in paths_map.items():
            arr = np.asarray(data)
            cum = np.cumprod(1 + arr, axis=1)
            median = np.median(cum, axis=0)
            composite += weight * median
        fig.add_trace(
            go.Scatter(
                x=months,
                y=composite / tot_w,
                mode="lines",
                name="Weighted",
                line=dict(dash="dash"),
            )
        )
    fig.update_layout(xaxis_title="Month", yaxis_title="Cumulative Return")
    return fig
