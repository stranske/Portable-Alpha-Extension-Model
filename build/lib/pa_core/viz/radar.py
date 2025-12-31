from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_metrics: pd.DataFrame) -> go.Figure:
    """Return radar chart comparing scenarios across metrics."""
    categories = list(df_metrics.columns)
    fig = go.Figure(layout_template=theme.TEMPLATE)
    for idx, row in df_metrics.iterrows():
        values = row.tolist()
        values += values[:1]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=str(idx),
            )
        )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
    return fig
