from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_grid: pd.DataFrame, *, param: str = "AE_leverage") -> go.Figure:
    """Return animated surface plot of parameter sweep."""
    unique = sorted(df_grid[param].unique())
    frames = []
    for val in unique:
        subset = df_grid[df_grid[param] == val]
        table = subset.pivot(index="ExtPA_frac", columns="AE_leverage", values="Sharpe")
        frames.append(
            go.Frame(
                data=[go.Surface(z=table.values, x=table.columns, y=table.index)],
                name=str(val),
            )
        )
    first = frames[0]
    fig = go.Figure(data=first.data, frames=frames, layout_template=theme.TEMPLATE)
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [{"label": "Play", "method": "animate", "args": [None]}],
            }
        ]
    )
    return fig
