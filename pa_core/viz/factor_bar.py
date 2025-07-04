from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df: pd.DataFrame) -> go.Figure:
    """Return grouped bar chart of factor exposures."""
    fig = go.Figure(layout_template=theme.TEMPLATE)
    palette = list(getattr(theme.TEMPLATE.layout, "colorway", []))
    if not palette:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    cat_colors: dict[str, str] = {}
    for idx, (agent, row) in enumerate(df.iterrows()):
        cat = theme.CATEGORY_BY_AGENT.get(agent, agent)
        if cat not in cat_colors:
            cat_colors[cat] = palette[len(cat_colors) % len(palette)]
        fig.add_trace(
            go.Bar(
                name=agent,
                x=list(df.columns),
                y=row.values,
                marker_color=cat_colors[cat],
            )
        )
    fig.update_layout(barmode="group", xaxis_title="Factor", yaxis_title="Exposure")
    return fig
