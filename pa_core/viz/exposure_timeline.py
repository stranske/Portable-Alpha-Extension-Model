from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(capital_by_month: pd.DataFrame) -> go.Figure:
    """Return stacked area chart of capital allocation over time."""
    df = capital_by_month.copy()
    fig = go.Figure(layout_template=theme.TEMPLATE)
    for i, col in enumerate(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                stackgroup="one",
                name=col,
            )
        )
    fig.update_layout(xaxis_title="Month", yaxis_title="Capital")
    return fig
