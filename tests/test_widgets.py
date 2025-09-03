import pandas as pd
import pytest

ipywidgets = pytest.importorskip("ipywidgets")
import sys
from pathlib import Path

pkg_path = Path(__file__).resolve().parent.parent / "pa_core" / "viz"
sys.path.insert(0, str(pkg_path.parent.parent))

import pa_core.viz.widgets as widgets
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
