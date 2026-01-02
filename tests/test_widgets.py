import pandas as pd
import pytest

import pa_core.viz.widgets as widgets

ipywidgets = pytest.importorskip("ipywidgets")


def test_explore_widget():
    df = pd.DataFrame(
        {
            "terminal_AnnReturn": [0.05],
            "monthly_AnnVol": [0.02],
            "TrackingErr": [0.01],
            "Agent": ["Base"],
            "terminal_ShortfallProb": [0.02],
        }
    )
    widgets.explore(df)
