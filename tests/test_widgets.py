import pandas as pd
import pytest

import pa_core.viz.widgets as widgets

ipywidgets = pytest.importorskip("ipywidgets")


def test_explore_widget():
    df = pd.DataFrame(
        {
            "AnnReturn": [0.05],
            "AnnVol": [0.02],
            "TrackingErr": [0.01],
            "Agent": ["Base"],
            "ShortfallProb": [0.02],
        }
    )
    widgets.explore(df)
