from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df: pd.DataFrame):
    """Return table visualisation.

    If Dash is available, return a ``dash_table.DataTable`` with CSV export.
    Otherwise fall back to a ``plotly.graph_objects.Figure`` table.
    """
    try:
        from dash import dash_table
    except Exception:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=list(df.columns)),
                    cells=dict(values=[df[c] for c in df.columns]),
                )
            ],
            layout_template=theme.TEMPLATE,
        )
        return fig

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        export_format="csv",
        style_table={"overflowX": "auto"},
    )
