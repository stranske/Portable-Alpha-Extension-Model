from __future__ import annotations

from collections.abc import Mapping

import plotly.graph_objects as go

from . import theme


def make(capital_map: Mapping[str, float]) -> go.Figure:
    """Return donut chart of capital by agent category."""
    grouped: dict[str, float] = {}
    for agent, capital in capital_map.items():
        cat = theme.CATEGORY_BY_AGENT.get(agent, agent)
        grouped[cat] = grouped.get(cat, 0.0) + float(capital)

    labels = list(grouped.keys())
    values = [grouped[c] for c in labels]

    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_pie(labels=labels, values=values, hole=0.4)
    fig.update_layout(title="Capital Allocation by Category")
    return fig
