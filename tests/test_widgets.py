import pandas as pd
import pytest

ipywidgets = pytest.importorskip("ipywidgets")
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

pkg_path = Path(__file__).resolve().parent.parent / "pa_core" / "viz"

sys.modules.setdefault(
    "pa_core", importlib.util.module_from_spec(importlib.machinery.ModuleSpec("pa_core", None))
)
viz_pkg = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("pa_core.viz", None))
viz_pkg.__path__ = [str(pkg_path)]
sys.modules["pa_core.viz"] = viz_pkg

spec = importlib.util.spec_from_file_location("pa_core.viz.widgets", pkg_path / "widgets.py")
widgets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(widgets)


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
