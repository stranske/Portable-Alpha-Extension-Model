from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df: pd.DataFrame) -> go.Figure:
    """Return parallel coordinates plot for multi-metric comparison."""
    dimensions = [dict(label=col, values=df[col]) for col in df.columns]
    fig = go.Figure(data=go.Parcoords(dimensions=dimensions), layout_template=theme.TEMPLATE)
    return fig
