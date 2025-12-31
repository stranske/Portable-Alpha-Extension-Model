from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(df_paths: pd.DataFrame | np.ndarray, bins: int = 10) -> go.Figure:
    """Return mosaic plot of final return distribution."""
    arr = np.asarray(df_paths)
    final = arr[:, -1]
    hist, edges = np.histogram(final, bins=bins, range=(final.min(), final.max()))
    side = int(np.ceil(np.sqrt(bins)))
    z = np.zeros((side, side))
    vals = np.repeat(hist, side * side // bins)[: side * side]
    z.flat[: len(vals)] = vals
    fig = go.Figure(go.Heatmap(z=z, showscale=False), layout_template=theme.TEMPLATE)
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    return fig
