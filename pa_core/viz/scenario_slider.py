from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from . import theme


def make(frames: Sequence[go.Frame]) -> go.Figure:
    """Return figure with slider to step through frames."""
    if not frames:
        return go.Figure()
    fig = go.Figure(frames=list(frames), layout_template=theme.TEMPLATE)
    data0 = getattr(frames[0], "data", None)
    if data0 and len(data0) > 0:
        fig.add_trace(data0[0])
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None]),
                ],
            )
        ],
        sliders=[
            {
                "steps": [
                    {"args": [[f.name], {}], "label": f.name, "method": "animate"}
                    for f in frames
                ]
            }
        ],
    )
    return fig
