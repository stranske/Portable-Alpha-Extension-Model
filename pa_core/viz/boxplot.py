from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    paths: Mapping[str, pd.DataFrame | np.ndarray] | pd.DataFrame | np.ndarray,
) -> go.Figure:
    """Return boxplot of monthly returns.

    Parameters
    ----------
    paths : mapping or array
        Either a mapping of agent name to return paths or a 2-D array.
    """
    if isinstance(paths, Mapping):
        data = {k: np.asarray(v).ravel() for k, v in paths.items()}
    else:
        arr = np.asarray(paths)
        data = {"Series": arr.ravel()}
    fig = go.Figure(layout_template=theme.TEMPLATE)
    for name, vals in data.items():
        fig.add_box(y=vals, name=str(name))
    fig.update_layout(yaxis_title="Monthly Return")
    return fig
