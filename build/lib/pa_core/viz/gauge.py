from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame, metric: str = "TrackingErr") -> go.Figure:
    """Return gauge of a risk metric against thresholds."""
    value = float(df_summary[metric].iloc[0])
    thr = theme.THRESHOLDS
    amber = thr.get("sharpe_amber", 0.4)
    green = thr.get("sharpe_green", 0.5)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            gauge={
                "axis": {"range": [None, green]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, amber], "color": "red"},
                    {"range": [amber, green], "color": "orange"},
                    {"range": [green, green * 1.2], "color": "green"},
                ],
            },
        )
    )
    fig.update_layout(template=theme.TEMPLATE)
    return fig
