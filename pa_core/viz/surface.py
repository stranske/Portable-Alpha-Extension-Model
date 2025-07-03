from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    df_grid: pd.DataFrame,
    *,
    x: str = "AE_leverage",
    y: str = "ExtPA_frac",
    z: str = "Sharpe",
) -> go.Figure:
    """Return 3-D surface plot of a parameter sweep."""
    table = (
        df_grid.pivot(index=y, columns=x, values=z)
        .sort_index()
        .sort_index(axis=1)
    )
    fig = go.Figure(
        data=go.Surface(z=table.values, x=table.columns, y=table.index),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
        )
    )
    return fig
