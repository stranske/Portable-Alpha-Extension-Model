from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import beta_scatter


def make(df_summary: pd.DataFrame, **kwargs) -> go.Figure:
    """Alias for :func:`beta_scatter.make`.

    Parameters
    ----------
    df_summary : pandas.DataFrame
        Summary table with TrackingErr, Beta and optional Capital columns.
    **kwargs
        Forwarded to ``beta_scatter.make``.
    """
    return beta_scatter.make(df_summary, **kwargs)
