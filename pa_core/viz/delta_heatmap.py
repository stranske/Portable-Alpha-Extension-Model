from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_a: pd.DataFrame, df_b: pd.DataFrame, *, value: str = "Sharpe") -> go.Figure:
    """Return heatmap of df_b minus df_a for ``value`` column."""
    diff = df_b.set_index(["AE_leverage", "ExtPA_frac"])[value] - df_a.set_index([
        "AE_leverage", "ExtPA_frac"]
    )[value]
    table = diff.unstack().sort_index().sort_index(axis=1)
    fig = go.Figure(
        data=go.Heatmap(z=table.values, x=table.columns, y=table.index),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(xaxis_title="AE_leverage", yaxis_title="ExtPA_frac")
    return fig
