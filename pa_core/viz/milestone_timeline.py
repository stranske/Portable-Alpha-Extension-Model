from __future__ import annotations

from collections.abc import Iterable

import plotly.graph_objects as go

from . import theme


def make(events: Iterable[tuple[int, str]], fig: go.Figure) -> go.Figure:
    """Annotate funding milestones on ``fig``."""
    out = go.Figure(fig)
    for month, label in events:
        out.add_vline(x=month, line_dash="dot", annotation_text=label)
    out.update_layout(template=theme.TEMPLATE)
    return out
