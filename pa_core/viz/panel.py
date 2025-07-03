from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme, risk_return, sharpe_ladder


def make(df_summary: pd.DataFrame, df_paths: pd.DataFrame | None = None) -> go.Figure:
    """Return composite panel with risk-return and Sharpe ladder charts."""
    fig = make_subplots(rows=1, cols=2)
    risk_fig = risk_return.make(df_summary)
    ladder_fig = sharpe_ladder.make(df_summary)
    for trace in risk_fig.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in ladder_fig.data:
        fig.add_trace(trace, row=1, col=2)
    fig.update_layout(template=theme.TEMPLATE)
    return fig

