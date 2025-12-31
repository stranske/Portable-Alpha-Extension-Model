from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import theme


def make(df: pd.DataFrame) -> go.Figure:
    """Return choropleth of regional exposure."""
    fig = px.choropleth(
        df,
        locations="Region",
        locationmode="country names",
        color="Exposure",
        color_continuous_scale=px.colors.sequential.Plasma,
    )
    fig.update_layout(template=theme.TEMPLATE)
    return fig
