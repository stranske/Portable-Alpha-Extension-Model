from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_errors: pd.DataFrame) -> go.Figure:
    """Return heatmap of data quality issues."""
    df = df_errors.fillna(0)
    fig = go.Figure(
        go.Heatmap(z=df.values, x=df.columns, y=df.index, colorscale="Reds"),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(xaxis_title="Field", yaxis_title="Date")
    return fig
