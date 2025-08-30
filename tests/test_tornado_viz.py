import importlib.util
import pathlib

import plotly.graph_objects as go

spec = importlib.util.spec_from_file_location(
    "tornado",
    pathlib.Path(__file__).resolve().parents[1] / "pa_core" / "viz" / "tornado.py",
)
tornado = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tornado)


def test_tornado_chart():
    contrib = {"A": 0.1, "B": -0.05, "C": 0.02}
    fig = tornado.make(contrib)
    assert isinstance(fig, go.Figure)
    fig.to_json()
