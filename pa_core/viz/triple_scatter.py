from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame) -> go.Figure:
    """Return 3-D scatter of TE, Beta and Excess Return."""
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(
        go.Scatter3d(
            x=df_summary["TrackingErr"],
            y=df_summary["Beta"],
            z=df_summary["AnnReturn"],
            mode="markers",
            text=df_summary.get("Agent"),
        )
    )
    fig.update_layout(
        scene=dict(xaxis_title="TE", yaxis_title="Beta", zaxis_title="ER")
    )
    return fig
