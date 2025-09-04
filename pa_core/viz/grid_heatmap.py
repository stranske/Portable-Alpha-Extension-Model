from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    df_grid: pd.DataFrame,
    *,
    x: str = "AE_leverage",
    y: str = "ExtPA_frac",
    z: str = "Sharpe",
    custom_fields: Optional[Iterable[str]] = None,
) -> go.Figure:
    """Return 2-D heatmap from a parameter grid.

    Parameters
    ----------
    df_grid: DataFrame with at least columns ``x``, ``y``, ``z``.
    x, y, z: Column names for axes and heat value.
    custom_fields: Optional iterable of additional column names to include as
        per-cell hover ``customdata``. Each field will be pivoted similarly to
        ``z`` and stacked along the last dimension.
    """
    table = (
        df_grid.pivot(index=y, columns=x, values=z).sort_index().sort_index(axis=1)
    )

    heatmap = go.Heatmap(z=table.values, x=table.columns, y=table.index)

    # Attach optional customdata for richer hover templates
    if custom_fields:
        stacks: List[np.ndarray] = []
        for field in custom_fields:
            if field in df_grid.columns:
                piv = (
                    df_grid.pivot(index=y, columns=x, values=field)
                    .reindex(index=table.index, columns=table.columns)
                    .values
                )
                stacks.append(piv)
        if stacks:
            # Shape: (ny, nx, n_fields)
            customdata = np.stack(stacks, axis=-1)
            heatmap.customdata = customdata  # type: ignore[assignment]

    fig = go.Figure(data=heatmap, layout_template=theme.TEMPLATE)
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig
