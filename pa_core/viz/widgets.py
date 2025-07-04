from __future__ import annotations

import pandas as pd
import ipywidgets as widgets
from IPython.display import display

from . import risk_return


def explore(df_summary: pd.DataFrame) -> None:
    """Display interactive risk-return explorer."""

    def _update(scale: float) -> None:
        scaled = df_summary.copy()
        scaled["AnnReturn"] *= scale
        fig = risk_return.make(scaled)
        display(fig)

    slider = widgets.FloatSlider(value=1.0, min=0.5, max=1.5, step=0.1)
    widgets.interact(_update, scale=slider)
