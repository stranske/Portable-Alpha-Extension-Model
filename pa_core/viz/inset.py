from __future__ import annotations

from typing import Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy

from . import theme


def make(fig: go.Figure, region: Tuple[float, float, float, float]) -> go.Figure:
    """Return figure with an inset zoomed on ``region``.

    Parameters
    ----------
    fig : go.Figure
        Base figure to embed the inset into.
    region : tuple
        (x0, x1, y0, y1) region to zoom in on.
    """
    x0, x1, y0, y1 = region
    out = make_subplots(insets=[{"cell": (1, 1), "l": 0.6, "b": 0.05, "w": 0.35, "h": 0.35}])
    for tr in fig.data:
        out.add_trace(tr)
        inset_trace = copy.deepcopy(tr)
        out.add_trace(inset_trace, row=1, col=1)
        out.data[-1].update(xaxis="x2", yaxis="y2", showlegend=False)
    out.update_xaxes(title=fig.layout.xaxis.title, row=1, col=1)
    out.update_yaxes(title=fig.layout.yaxis.title, row=1, col=1)
    out.layout["xaxis2"].update(range=[x0, x1])
    out.layout["yaxis2"].update(range=[y0, y1])
    out.update_layout(template=theme.TEMPLATE)
    return out
