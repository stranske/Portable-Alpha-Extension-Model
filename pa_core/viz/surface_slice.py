from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_grid: pd.DataFrame, axis: str = "AE_leverage") -> go.Figure:
    """Return heatmap slice of 3-D surface."""
    vals = sorted(df_grid[axis].unique())
    first = vals[0]
    table = df_grid[df_grid[axis] == first].pivot(
        index="ExtPA_frac", columns="AE_leverage", values="Sharpe"
    )
    fig = go.Figure(
        data=go.Heatmap(z=table.values, x=table.columns, y=table.index),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(xaxis_title=table.columns.name, yaxis_title=table.index.name)
    return fig
