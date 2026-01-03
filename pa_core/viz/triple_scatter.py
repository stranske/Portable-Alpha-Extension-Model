from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame) -> go.Figure:
    """Return 3-D scatter of monthly_TE, Beta and terminal_AnnReturn."""
    x_col = "monthly_TE" if "monthly_TE" in df_summary else "TrackingErr"
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(
        go.Scatter3d(
            x=df_summary[x_col],
            y=df_summary["Beta"],
            z=df_summary["terminal_AnnReturn"],
            mode="markers",
            text=df_summary.get("Agent"),
        )
    )
    fig.update_layout(
        scene=dict(xaxis_title=x_col, yaxis_title="Beta", zaxis_title="terminal_AnnReturn")
    )
    return fig
