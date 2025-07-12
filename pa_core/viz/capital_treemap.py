from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    capital: dict[str, float], category_map: dict[str, str] | None = None
) -> go.Figure:
    """Return treemap of capital allocation by agent category."""
    df = pd.DataFrame(
        {"Agent": list(capital.keys()), "Capital": list(capital.values())}
    )
    if category_map is None:
        cat = {a: theme.CATEGORY_BY_AGENT.get(a, a) for a in df["Agent"]}
    else:
        cat = {
            a: category_map.get(a, theme.CATEGORY_BY_AGENT.get(a, a))
            for a in df["Agent"]
        }
    df["Category"] = df["Agent"].map(lambda x: cat.get(x))
    fig = go.Figure(
        go.Treemap(labels=df["Agent"], parents=df["Category"], values=df["Capital"]),
        layout_template=theme.TEMPLATE,
    )
    return fig
