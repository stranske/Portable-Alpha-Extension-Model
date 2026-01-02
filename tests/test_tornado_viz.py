import pandas as pd
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
    assert fig.layout.xaxis.title.text == "Delta (terminal_AnnReturn, %)"
    assert fig.layout.xaxis.tickformat == ".2%"
    y_vals = list(fig.data[0].y)
    assert y_vals == ["A", "B", "C"]


def test_tornado_series_from_sensitivity_uses_attrs():
    sens_df = pd.DataFrame({"Parameter": ["B", "A", "C"], "DeltaAbs": [0.02, 0.02, 0.01]})
    sens_df.attrs.update({"metric": "Sharpe", "units": "x"})
    series = tornado.series_from_sensitivity(sens_df)
    assert list(series.index) == ["A", "B", "C"]
    fig = tornado.make(series)
    assert fig.layout.xaxis.title.text == "Delta (Sharpe, x)"
