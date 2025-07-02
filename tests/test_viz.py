import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pa_core.viz import risk_return, fan, path_dist, sharpe_ladder


def test_risk_return_make():
    df = pd.DataFrame({
        "AnnReturn": [0.05],
        "AnnVol": [0.02],
        "TrackingErr": [0.01],
        "Agent": ["Base"],
        "ShortfallProb": [0.02],
    })
    fig = risk_return.make(df)
    assert isinstance(fig, go.Figure)
    fig.to_json()


def test_fan_and_dist():
    arr = np.random.normal(size=(10, 5))
    fig1 = fan.make(arr)
    fig2 = path_dist.make(arr)
    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)
    fig1.to_json()
    fig2.to_json()


def test_sharpe_ladder():
    df = pd.DataFrame({
        "AnnReturn": [0.05, 0.03],
        "AnnVol": [0.02, 0.04],
        "Agent": ["A", "B"],
    })
    fig = sharpe_ladder.make(df)
    assert isinstance(fig, go.Figure)
    fig.to_json()
