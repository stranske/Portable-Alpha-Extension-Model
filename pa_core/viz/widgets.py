from __future__ import annotations

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from . import risk_return


def explore(df_summary: pd.DataFrame) -> None:
    """Display interactive risk-return explorer."""

    def _update(scale: float) -> None:
        scaled = df_summary.copy()
        scaled["AnnReturn"] *= scale
        fig = risk_return.make(scaled)
        display(fig)  # type: ignore[no-untyped-call]

    slider = widgets.FloatSlider(value=1.0, min=0.5, max=1.5, step=0.1)
    widgets.interact(_update, scale=slider)
