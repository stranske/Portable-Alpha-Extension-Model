from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_paths: pd.DataFrame | np.ndarray, *, by_month: bool = False) -> go.Figure:
    """Return violin plot of monthly returns.

    Parameters
    ----------
    df_paths : pandas.DataFrame or numpy.ndarray
        Simulation paths with shape (n_sim, n_months).
    by_month : bool, optional
        If True, draw a violin per month; otherwise aggregate all months.
    """
    arr = np.asarray(df_paths)
    fig = go.Figure(layout_template=theme.TEMPLATE)

    if by_month:
        for i in range(arr.shape[1]):
            fig.add_trace(
                go.Violin(
                    y=arr[:, i],
                    name=f"Month {i + 1}",
                    box_visible=True,
                    meanline_visible=True,
                )
            )
    else:
        fig.add_trace(
            go.Violin(
                y=arr.ravel(),
                box_visible=True,
                meanline_visible=True,
                name="Returns",
            )
        )

    fig.update_layout(yaxis_title="Monthly Return")
    return fig
