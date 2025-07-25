from __future__ import annotations

from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from . import theme


def make(paths_map: Mapping[str, pd.DataFrame | np.ndarray]) -> go.Figure:
    """Return overlay of median cumulative return paths.

    Colours are assigned by ``theme.CATEGORY_BY_AGENT`` so that multiple
    charts share a consistent palette.
    """
    fig = go.Figure(layout_template=theme.TEMPLATE)

    # Build deterministic colour mapping by category
    tmpl = theme.TEMPLATE
    if isinstance(tmpl, str):
        tmpl = pio.templates[tmpl]

    palette = None
    layout = cast(Any, getattr(tmpl, "layout", None))
    if layout is not None and not isinstance(layout, tuple):
        palette = getattr(layout, "colorway", None)
    if palette is None:
        palette = cast(Any, pio.templates["plotly"].layout).colorway
    palette = list(palette)
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
