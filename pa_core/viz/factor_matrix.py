from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df: pd.DataFrame) -> go.Figure:
    """Return heatmap of factor sensitivities."""
    fig = go.Figure(
        data=go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index)),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Factor")
    return fig
