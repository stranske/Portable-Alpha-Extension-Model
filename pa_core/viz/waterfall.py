from __future__ import annotations

from typing import Mapping

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(contrib: Mapping[str, float] | pd.Series) -> go.Figure:
    """Return waterfall chart of risk or return contributions."""
    if isinstance(contrib, pd.Series):
        series: pd.Series = contrib
    else:
        series = pd.Series({str(k): float(v) for k, v in contrib.items()})
    labels = series.index.tolist()
    values = series.astype(float).tolist()
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(
        go.Waterfall(x=labels, y=values, connector=dict(line=dict(color="grey")))
    )
    fig.update_layout(xaxis_title="Agent", yaxis_title="Contribution")
    return fig
