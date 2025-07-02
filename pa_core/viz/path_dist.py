from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_paths: pd.DataFrame | np.ndarray) -> go.Figure:
    """Return histogram of end-point returns."""
    arr = np.asarray(df_paths)
    final = arr[:, -1].ravel()
    fig = go.Figure(layout_template=theme.TEMPLATE)
    fig.add_trace(go.Histogram(x=final, nbinsx=40, name="Distribution"))
    fig.update_layout(xaxis_title="Final Return", yaxis_title="Frequency")
    return fig
