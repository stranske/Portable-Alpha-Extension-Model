from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(beta_by_month: pd.DataFrame) -> go.Figure:
    """Return heatmap of beta exposure over time."""
    fig = go.Figure(
        data=go.Heatmap(
            z=beta_by_month.values, x=beta_by_month.index, y=beta_by_month.columns
        ),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Agent")
    return fig
