from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(paths_map: Mapping[str, pd.DataFrame | np.ndarray]) -> go.Figure:
    """Return overlay of median cumulative return paths.

    Colours are assigned by ``theme.CATEGORY_BY_AGENT`` so that multiple
    charts share a consistent palette.
    """
    fig = go.Figure(layout_template=theme.TEMPLATE)

    # Build deterministic colour mapping by category
    palette = list(theme.TEMPLATE.layout.colorway)
    cat_to_color: dict[str, str] = {}

    for name, data in paths_map.items():
        arr = np.asarray(data)
        cum = np.cumprod(1 + arr, axis=1)
        median = np.median(cum, axis=0)
        months = np.arange(median.size)

        cat = theme.CATEGORY_BY_AGENT.get(name, name)
        if cat not in cat_to_color:
            color = palette[len(cat_to_color) % len(palette)]
            cat_to_color[cat] = color

        fig.add_trace(
            go.Scatter(
                x=months,
                y=median,
                mode="lines",
                name=name,
                line=dict(color=cat_to_color[cat]),
            )
        )

    fig.update_layout(xaxis_title="Month", yaxis_title="Cumulative Return")
    return fig
