from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme


def make(df_series: pd.DataFrame) -> go.Figure:
    """Return matrix of sparkline charts."""
    cols = list(df_series.columns)
    rows = len(cols)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True)
    for i, col in enumerate(cols, start=1):
        fig.add_trace(
            go.Scatter(
                x=df_series.index, y=df_series[col], mode="lines", name=str(col)
            ),
            row=i,
            col=1,
        )
    fig.update_layout(template=theme.TEMPLATE, height=80 * rows)
    for i in range(rows):
        fig.update_yaxes(showticklabels=False, row=i + 1, col=1)
    return fig
