from __future__ import annotations

from collections.abc import Iterable

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme


def make(figures: Iterable[go.Figure], cols: int = 2) -> go.Figure:
    """Arrange multiple figures on one canvas."""
    figs = list(figures)
    rows = (len(figs) + cols - 1) // cols
    out = make_subplots(rows=rows, cols=cols)
    r = c = 1
    for fig in figs:
        for tr in fig.data:
            out.add_trace(tr, row=r, col=c)
        c += 1
        if c > cols:
            c = 1
            r += 1
    out.update_layout(template=theme.TEMPLATE)
    return out
