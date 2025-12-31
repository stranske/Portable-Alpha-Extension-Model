from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df: pd.DataFrame) -> go.Figure:
    """Return stacked line chart of factor exposures over time."""
    df = df.copy()
    fig = go.Figure(layout_template=theme.TEMPLATE)
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)))
    fig.update_layout(xaxis_title="Month", yaxis_title="Exposure")
    return fig
