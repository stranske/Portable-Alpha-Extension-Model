import plotly.graph_objects as go

from pa_core.viz import tornado


def test_tornado_chart():
    contrib = {"A": 0.1, "B": -0.05, "C": 0.02}
    fig = tornado.make(contrib)
    assert isinstance(fig, go.Figure)
    fig.to_json()
