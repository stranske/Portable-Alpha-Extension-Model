from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(paths: dict[str, pd.DataFrame | np.ndarray], *, threshold: float = 0.3) -> go.Figure:
    """Return correlation network diagram."""
    names = list(paths.keys())
    series = [np.asarray(v).mean(axis=0) for v in paths.values()]
    corr = np.corrcoef(series)
    n = len(names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="markers+text", text=names, textposition="bottom center")
    )
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] >= threshold:
                fig.add_trace(
                    go.Scatter(
                        x=[xs[i], xs[j]],
                        y=[ys[i], ys[j]],
                        mode="lines",
                        line=dict(width=1),
                        showlegend=False,
                    )
                )
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    return fig
