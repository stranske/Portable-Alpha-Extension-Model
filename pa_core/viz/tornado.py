from __future__ import annotations

from typing import Mapping, cast

import pandas as pd
import plotly.graph_objects as go

from . import theme

__all__ = ["make"]


def make(
    contrib: Mapping[str, float] | pd.Series, title: str | None = None
) -> go.Figure:
    """Render a tornado chart from a mapping/Series of contributions.

    Inputs may be a dict-like mapping of parameter -> delta contribution,
    or a pandas Series. Bars are sorted by absolute value descending.
    """
    if isinstance(contrib, pd.Series):
        series = cast(pd.Series, contrib)
    else:
        series = pd.Series({str(k): float(v) for k, v in contrib.items()})

    # Ensure numeric dtype for plotting
    series = series.astype(float)

    if series.empty:
        return go.Figure(layout_template=theme.TEMPLATE)

    series = series.reindex(series.abs().sort_values(ascending=False).index)
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(go.Bar(x=series.values, y=series.index.tolist(), orientation="h"))
    fig.update_layout(
        title=title or "Sensitivity Tornado",
        xaxis_title="Delta",
        yaxis_title="Parameter",
    )
    return fig
