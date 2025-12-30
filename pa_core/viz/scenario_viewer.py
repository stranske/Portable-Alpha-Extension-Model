from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import overlay, scenario_slider, theme


def make(paths_map: Mapping[str, pd.DataFrame | np.ndarray]) -> go.Figure:
    """Return figure combining overlay and scenario slider."""
    # Overlay traces
    over_fig = overlay.make(paths_map)

    # Build frames from first agent's paths
    first_data = next(iter(paths_map.values()))
    arr = np.asarray(first_data)
    months = np.arange(arr.shape[1])
    frames = [
        go.Frame(
            data=[go.Scatter(x=months, y=np.cumprod(1 + arr[i], axis=0))],
            name=str(i),
        )
        for i in range(min(len(arr), 10))
    ]
    slider_fig = scenario_slider.make(frames)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Overlay", "Scenarios"))
    for tr in over_fig.data:
        fig.add_trace(tr, row=1, col=1)
    if slider_fig.data:
        for tr in slider_fig.data:
            fig.add_trace(tr, row=1, col=2)
    fig.frames = slider_fig.frames
    fig.update_layout(
        template=theme.TEMPLATE,
        updatemenus=slider_fig.layout.updatemenus,
        sliders=slider_fig.layout.sliders,
        xaxis_title="Month",
        xaxis2_title="Month",
    )
    return fig
