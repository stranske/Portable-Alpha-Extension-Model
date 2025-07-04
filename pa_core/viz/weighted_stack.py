from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df: pd.DataFrame) -> go.Figure:
    """Return weighted stacked bar chart."""
    fig = go.Figure(layout_template=theme.TEMPLATE)
    widths = df.index.to_numpy()
    for col in df.columns:
        fig.add_bar(x=widths, y=df[col], name=str(col))
    fig.update_layout(barmode="stack", xaxis_title="Horizon", yaxis_title="Value")
    return fig
