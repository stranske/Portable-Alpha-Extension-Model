from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme


def make(figs: Iterable[go.Figure], df: pd.DataFrame) -> go.Figure:
    """Return composite figure with linked x-range.

    This is a lightweight placeholder implementation that simply arranges the
    provided figures in a column and applies a shared hovermode.
    """
    figs = list(figs)
    if not figs:
        return go.Figure()

    out = make_subplots(rows=len(figs), cols=1, shared_xaxes=True)
    for i, fig in enumerate(figs, start=1):
        for trace in fig.data:
            out.add_trace(trace, row=i, col=1)
    out.update_layout(template=theme.TEMPLATE, hovermode="x unified")
    return out
