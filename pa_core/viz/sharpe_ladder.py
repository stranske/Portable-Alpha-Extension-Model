from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_summary: pd.DataFrame) -> go.Figure:
    """Return bar chart of Sharpe ratio sorted descending."""
    df = df_summary.copy()
    df["Sharpe"] = df["AnnReturn"] / df["AnnVol"].replace(0, pd.NA)
    df = df.sort_values("Sharpe", ascending=False)
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_bar(x=df["Agent"], y=df["Sharpe"])
    fig.update_layout(xaxis_title="Agent", yaxis_title="Sharpe Ratio")
    return fig
