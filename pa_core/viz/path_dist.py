from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_paths: pd.DataFrame | np.ndarray) -> go.Figure:
    """Return histogram of end-point returns with optional CDF view."""
    arr = np.asarray(df_paths)
    final = arr[:, -1].ravel()

    hist = go.Histogram(x=final, nbinsx=40, name="Histogram")
    cdf = go.Scatter(
        x=np.sort(final),
        y=np.linspace(0, 1, final.size),
        mode="lines",
        name="CDF",
    )

    fig = go.Figure(data=[hist, cdf], layout_template=theme.TEMPLATE)
    fig.data[1].visible = False
    fig.update_layout(
        xaxis_title="Final Return",
        yaxis_title="Frequency",
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="Histogram",
                        method="update",
                        args=[{"visible": [True, False]}, {"yaxis": {"title": "Frequency"}}],
                    ),
                    dict(
                        label="CDF",
                        method="update",
                        args=[{"visible": [False, True]}, {"yaxis": {"title": "Cumulative Probability"}}],
                    ),
                ],
                direction="left",
                showactive=True,
                x=0.5,
                y=1.1,
                xanchor="center",
                yanchor="top",
            )
        ],
    )
    return fig
