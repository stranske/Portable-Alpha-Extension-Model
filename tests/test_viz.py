import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pa_core.viz import (
    risk_return,
    fan,
    path_dist,
    sharpe_ladder,
    rolling_panel,
    surface,
    pptx_export,
    html_export,
    corr_heatmap,
    category_pie,
    animation,
    overlay,
    waterfall,
    export_bundle,
    data_table,
    scenario_viewer,
)


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


def test_rolling_panel_and_surface_and_pptx(tmp_path):
    arr = np.random.normal(size=(20, 24))
    panel_fig = rolling_panel.make(arr)
    assert isinstance(panel_fig, go.Figure)
    panel_fig.to_json()

    grid = pd.DataFrame({
        "AE_leverage": [1, 1, 2, 2],
        "ExtPA_frac": [0.2, 0.4, 0.2, 0.4],
        "Sharpe": [0.4, 0.5, 0.55, 0.6],
    })
    surf_fig = surface.make(grid)
    assert isinstance(surf_fig, go.Figure)
    surf_fig.to_json()

    out = tmp_path / "out.pptx"
    pptx_export.save([panel_fig, surf_fig], out)
    assert out.exists()


def test_category_pie():
    fig = category_pie.make({
        "BaseAgent": 500,
        "ExternalPAAgent": 300,
        "InternalPAAgent": 200,
    })
    assert isinstance(fig, go.Figure)
    fig.to_json()


def test_html_and_corr(tmp_path):
    fig = risk_return.make(
        pd.DataFrame({
            "AnnReturn": [0.05],
            "AnnVol": [0.02],
            "TrackingErr": [0.01],
            "Agent": ["Base"],
            "ShortfallProb": [0.02],
        })
    )
    out_html = tmp_path / "fig.html"
    html_export.save(fig, out_html)
    assert out_html.exists()

    arr = np.random.normal(size=(5, 3))
    fig2 = corr_heatmap.make({"A": arr, "B": arr})
    assert isinstance(fig2, go.Figure)
    fig2.to_json()


def test_animation():
    arr = np.random.normal(size=(10, 6))
    fig = animation.make(arr)
    assert isinstance(fig, go.Figure)
    fig.to_json()


def test_overlay_and_waterfall_and_bundle(tmp_path):
    arr = np.random.normal(size=(5, 4))
    over_fig = overlay.make({"A": arr, "B": arr})
    assert isinstance(over_fig, go.Figure)
    over_fig.to_json()

    wf_fig = waterfall.make({"A": 0.1, "B": -0.05})
    assert isinstance(wf_fig, go.Figure)
    wf_fig.to_json()

    export_bundle.save([over_fig, wf_fig], tmp_path / "bundle")
    assert (tmp_path / "bundle_1.html").exists()


def test_data_table_and_scenario_viewer():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    table_obj = data_table.make(df)
    assert table_obj is not None
    arr = np.random.normal(size=(3, 6))
    fig = scenario_viewer.make({"A": arr})
    assert isinstance(fig, go.Figure)
    fig.to_json()
