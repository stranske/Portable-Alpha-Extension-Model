from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import theme


def make(df_paths: pd.DataFrame) -> go.Figure:
    """Return heatmap of monthly returns by year and month."""
    df = df_paths.copy()
    df = df.unstack(0)
    fig = px.imshow(df, aspect="auto", color_continuous_scale="RdBu_r")
    fig.update_layout(template=theme.TEMPLATE)
    return fig
