import plotly.graph_objects as go

from pa_core.viz import tornado


def test_tornado_chart():
    contrib = {"A": 0.1, "B": -0.05, "C": 0.02}
    fig = tornado.make(contrib)
    assert isinstance(fig, go.Figure)
    fig.to_json()


def test_tornado_chart_orders_ties_and_labels_units():
    contrib = {"B": 0.1, "A": -0.1, "C": 0.05}
    fig = tornado.make(contrib)
    assert fig.layout.xaxis.title.text == "Delta (AnnReturn, %)"
    y_vals = list(fig.data[0].y)
    assert y_vals == ["A", "B", "C"]
