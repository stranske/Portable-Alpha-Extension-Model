from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme


def make(
    regime_labels: pd.DataFrame | np.ndarray, label_order: Sequence[object] | None = None
) -> go.Figure:
    """Return stacked area chart of regime probabilities over time."""
    df = _coerce_regime_df(regime_labels)
    labels = _resolve_label_order(df, label_order)
    x_vals = _resolve_x_values(df)
    fig = go.Figure(layout_template=theme.TEMPLATE)
    probs = _regime_probabilities(df, labels)
    for label in labels:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=probs[label],
                mode="lines",
                name=str(label),
                stackgroup="one",
            )
        )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Regime Probability",
        yaxis_tickformat=".0%",
    )
    return fig


def _coerce_regime_df(regime_labels: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(regime_labels, pd.DataFrame):
        df = regime_labels.copy()
    else:
        arr = np.asarray(regime_labels)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("regime_labels must be 1D or 2D")
        df = pd.DataFrame(arr)
    if df.empty:
        raise ValueError("regime_labels must not be empty")
    return df


def _resolve_label_order(df: pd.DataFrame, label_order: Sequence[object] | None) -> list[object]:
    discovered = list(pd.unique(df.to_numpy().ravel()))
    if label_order is None:
        return discovered
    resolved = list(label_order)
    for label in discovered:
        if label not in resolved:
            resolved.append(label)
    return resolved


def _resolve_x_values(df: pd.DataFrame) -> Iterable[object]:
    if df.columns is None or len(df.columns) == 0:
        return range(df.shape[1])
    return df.columns


def _regime_probabilities(df: pd.DataFrame, labels: Sequence[object]) -> dict[object, np.ndarray]:
    n_sim = df.shape[0]
    if n_sim <= 0:
        raise ValueError("regime_labels must contain at least one simulation")
    probs: dict[object, np.ndarray] = {}
    for label in labels:
        counts = df.eq(label).sum(axis=0).to_numpy(dtype=float)
        probs[label] = counts / float(n_sim)
    return probs
