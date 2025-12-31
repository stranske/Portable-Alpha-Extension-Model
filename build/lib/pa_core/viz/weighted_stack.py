from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df: pd.DataFrame) -> go.Figure:
    """Return weighted stacked bar chart.

    Bar width is proportional to the horizon length (index value) so
    longer windows visually dominate shorter ones.
    """
    horizons = np.asarray(df.index, dtype=float)
    # Position bars by cumulative width so they do not overlap
    edges = np.concatenate([[0.0], np.cumsum(horizons)])
    x_pos = 0.5 * (edges[:-1] + edges[1:])

    fig = go.Figure(layout_template=theme.TEMPLATE)
    for col in df.columns:
        fig.add_bar(x=x_pos, y=df[col], width=horizons, name=str(col))
    fig.update_layout(
        barmode="stack",
        xaxis_title="Horizon",
        yaxis_title="Value",
        xaxis=dict(tickvals=x_pos, ticktext=horizons.astype(str)),
    )
    return fig
