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
    grid_heatmap,
    panel,
    violin,
    rolling_corr_heatmap,
    exposure_timeline,
    gauge,
    radar,
    scatter_matrix,
    risk_return_bubble,
    rolling_var,
    breach_calendar,
    moments_panel,
    parallel_coords,
    capital_treemap,
    corr_network,
    beta_heatmap,
    beta_scatter,
    overlay_weighted,
    factor_bar,
    factor_matrix,
    multi_fan,
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


def test_data_table_and_scenario_viewer_and_heatmap():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    table_obj = data_table.make(df)
    assert table_obj is not None
    arr = np.random.normal(size=(3, 6))
    fig = scenario_viewer.make({"A": arr})
    assert isinstance(fig, go.Figure)
    fig.to_json()

    grid = pd.DataFrame({
        "AE_leverage": [1, 2],
        "ExtPA_frac": [0.2, 0.4],
        "Sharpe": [0.5, 0.6],
    })
    heatmap_fig = grid_heatmap.make(grid)
    assert isinstance(heatmap_fig, go.Figure)
    heatmap_fig.to_json()


def test_panel_make():
    df = pd.DataFrame({
        "AnnReturn": [0.05, 0.04],
        "AnnVol": [0.02, 0.03],
        "TrackingErr": [0.01, 0.015],
        "Agent": ["A", "B"],
        "ShortfallProb": [0.02, 0.03],
    })
    fig = panel.make(df)
    assert isinstance(fig, go.Figure)
    fig.to_json()


def test_violin_make():
    arr = np.random.normal(size=(10, 5))
    fig = violin.make(arr, by_month=True)
    assert isinstance(fig, go.Figure)
    fig.to_json()


def test_new_viz_helpers():
    arr = np.random.normal(size=(10, 12))
    heat_fig = rolling_corr_heatmap.make(arr, window=3)
    assert isinstance(heat_fig, go.Figure)
    heat_fig.to_json()

    df_cap = pd.DataFrame({
        "A": [100, 120, 110],
        "B": [50, 60, 55],
    }, index=[0, 1, 2])
    exp_fig = exposure_timeline.make(df_cap)
    assert isinstance(exp_fig, go.Figure)
    exp_fig.to_json()

    summary = pd.DataFrame({"TrackingErr": [0.02]})
    gauge_fig = gauge.make(summary)
    assert isinstance(gauge_fig, go.Figure)
    gauge_fig.to_json()

    metrics = pd.DataFrame({
        "TE": [0.02, 0.03],
        "ER": [0.05, 0.06],
    }, index=["Scenario1", "Scenario2"])
    radar_fig = radar.make(metrics)
    assert isinstance(radar_fig, go.Figure)
    radar_fig.to_json()


def test_extra_viz_helpers():
    df = pd.DataFrame({
        "AnnReturn": [0.05, 0.04],
        "AnnVol": [0.02, 0.03],
        "TrackingErr": [0.01, 0.015],
        "Capital": [100, 200],
        "ShortfallProb": [0.02, 0.03],
    })
    bubble_fig = risk_return_bubble.make(df)
    assert isinstance(bubble_fig, go.Figure)
    bubble_fig.to_json()

    arr = np.random.normal(size=(10, 12))
    var_fig = rolling_var.make(arr, window=3)
    assert isinstance(var_fig, go.Figure)
    var_fig.to_json()

    summary = pd.DataFrame({
        "Month": [0, 1, 2],
        "TrackingErr": [0.02, 0.04, 0.01],
        "ShortfallProb": [0.01, 0.15, 0.02],
    })
    breach_fig = breach_calendar.make(summary)
    assert isinstance(breach_fig, go.Figure)
    breach_fig.to_json()

    sm_fig = scatter_matrix.make(df[["AnnReturn", "AnnVol", "TrackingErr"]])
    assert isinstance(sm_fig, go.Figure)
    sm_fig.to_json()

    moments_fig = moments_panel.make(arr)
    assert isinstance(moments_fig, go.Figure)
    moments_fig.to_json()

    pc_fig = parallel_coords.make(df[["AnnReturn", "AnnVol", "TrackingErr"]])
    assert isinstance(pc_fig, go.Figure)
    pc_fig.to_json()


def test_newly_added_viz_helpers():
    cap_fig = capital_treemap.make({"A": 100, "B": 200})
    assert isinstance(cap_fig, go.Figure)
    cap_fig.to_json()

    arr = np.random.normal(size=(5, 4))
    net_fig = corr_network.make({"A": arr, "B": arr}, threshold=0.1)
    assert isinstance(net_fig, go.Figure)
    net_fig.to_json()

    beta_df = pd.DataFrame({"A": [1, 2], "B": [0.5, 0.6]}, index=[0, 1])
    beta_fig = beta_heatmap.make(beta_df)
    assert isinstance(beta_fig, go.Figure)
    beta_fig.to_json()


def test_additional_new_viz_helpers():
    df = pd.DataFrame({
        "TrackingErr": [0.02, 0.03],
        "Beta": [1.1, 0.9],
        "Capital": [100, 200],
        "ShortfallProb": [0.01, 0.2],
        "Agent": ["A", "B"],
    })
    bs_fig = beta_scatter.make(df)
    assert isinstance(bs_fig, go.Figure)
    bs_fig.to_json()

    arr = np.random.normal(size=(4, 6))
    ow_fig = overlay_weighted.make({"A": (arr, 1.0)})
    assert isinstance(ow_fig, go.Figure)
    ow_fig.to_json()

    exp_df = pd.DataFrame({"Val": [0.1, 0.2], "Mom": [0.3, -0.1]}, index=["A", "B"])
    fb_fig = factor_bar.make(exp_df)
    assert isinstance(fb_fig, go.Figure)
    fb_fig.to_json()

    fm_fig = factor_matrix.make(exp_df.T)
    assert isinstance(fm_fig, go.Figure)
    fm_fig.to_json()

    mf_fig = multi_fan.make(arr, horizons=[3, 5])
    assert isinstance(mf_fig, go.Figure)
    mf_fig.to_json()


