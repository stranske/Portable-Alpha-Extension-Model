from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_returns: pd.DataFrame) -> go.Figure:
    """Return sunburst chart of return attribution by sleeve."""
    df = df_returns.copy()
    if {"Agent", "Sub", "Return"} <= set(df.columns):
        df["Category"] = df["Agent"].map(lambda a: theme.CATEGORY_BY_AGENT.get(a, a))
        rows: list[tuple[str, str, float]] = []
        for cat, cat_df in df.groupby("Category"):
            cat_ret = float(cat_df["Return"].sum())
            rows.append((str(cat), "", cat_ret))
            for agent, ag_df in cat_df.groupby("Agent"):
                ag_ret = float(ag_df["Return"].sum())
                rows.append((str(agent), str(cat), ag_ret))
                for _, row in ag_df.iterrows():
                    rows.append((str(row["Sub"]), str(agent), float(row["Return"])) )
    else:
        raise ValueError("DataFrame must contain Agent, Sub and Return columns")

    labels = [r[0] for r in rows]
    parents = [r[1] for r in rows]
    values = [r[2] for r in rows]

    fig = go.Figure(
        go.Sunburst(labels=labels, parents=parents, values=values),
        layout_template=theme.TEMPLATE,
    )
    return fig
