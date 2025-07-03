from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(paths_map: dict[str, pd.DataFrame | np.ndarray]) -> go.Figure:
    """Return heatmap of monthly return correlations."""
    arrays = [np.asarray(v) for v in paths_map.values()]
    # flatten along sims
    monthly = [arr.reshape(-1, arr.shape[-1]) for arr in arrays]
    returns = np.concatenate(monthly, axis=0)
    corr = np.corrcoef(returns.T)
    idx = list(range(corr.shape[0]))
    fig = go.Figure(
        data=go.Heatmap(z=corr, x=idx, y=idx),
        layout_template=theme.TEMPLATE,
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Month")
    return fig
