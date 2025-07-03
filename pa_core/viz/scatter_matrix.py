from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import theme


def make(df: pd.DataFrame) -> go.Figure:
    """Return scatter-matrix plot coloured by agent category."""
    fig = px.scatter_matrix(df, color_discrete_sequence=theme.TEMPLATE.layout.colorway)  # type: ignore[attr-defined]
    fig.update_layout(template=theme.TEMPLATE)
    return fig
