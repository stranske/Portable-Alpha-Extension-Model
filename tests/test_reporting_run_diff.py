from __future__ import annotations

import pandas as pd
import pytest

from pa_core.reporting.run_diff import build_run_diff


def test_build_run_diff_config_and_metrics() -> None:
    current_manifest = {"config": {"a": 1, "b": 2}}
    previous_manifest = {"config": {"a": 1, "b": 1, "c": 3}}

    current_summary = pd.DataFrame(
        {"AnnReturn": [0.05], "AnnVol": [0.1], "Label": ["current"]}
    )
    previous_summary = pd.DataFrame(
        {"AnnReturn": [0.04], "AnnVol": [0.12], "Label": ["previous"]}
    )

    cfg_diff, metric_diff = build_run_diff(
        current_manifest, previous_manifest, current_summary, previous_summary
    )

    assert set(cfg_diff["Parameter"]) == {"b", "c"}
    b_row = cfg_diff[cfg_diff["Parameter"] == "b"].iloc[0]
    assert b_row["Current"] == 2
    assert b_row["Previous"] == 1
    assert b_row["Delta"] == pytest.approx(1.0)
    c_row = cfg_diff[cfg_diff["Parameter"] == "c"].iloc[0]
    assert c_row["Delta"] == ""

    assert set(metric_diff["Metric"]) == {"AnnReturn", "AnnVol"}
    ann_return = metric_diff[metric_diff["Metric"] == "AnnReturn"].iloc[0]
    assert ann_return["Delta"] == pytest.approx(0.01)


def test_build_run_diff_handles_empty_summaries() -> None:
    empty = pd.DataFrame()
    cfg_diff, metric_diff = build_run_diff(None, None, empty, empty)

    assert cfg_diff.empty
    assert metric_diff.empty


def test_build_run_diff_aligns_by_agent() -> None:
    current_summary = pd.DataFrame(
        {
            "Agent": ["Base", "Alt"],
            "AnnReturn": [0.05, 0.03],
            "AnnVol": [0.1, 0.2],
        }
    )
    previous_summary = pd.DataFrame(
        {
            "Agent": ["Base", "Alt"],
            "AnnReturn": [0.04, 0.02],
            "AnnVol": [0.11, 0.19],
        }
    )

    _, metric_diff = build_run_diff({}, {}, current_summary, previous_summary)

    assert "Agent" in metric_diff.columns
    base_rows = metric_diff[metric_diff["Agent"] == "Base"]
    assert set(base_rows["Metric"]) >= {"AnnReturn", "AnnVol"}


def test_build_run_diff_aligns_by_combination_and_agent() -> None:
    current_summary = pd.DataFrame(
        {
            "Agent": ["Base", "Base"],
            "Combination": ["Run1", "Run2"],
            "AnnReturn": [0.05, 0.04],
            "AnnVol": [0.1, 0.12],
        }
    )
    previous_summary = pd.DataFrame(
        {
            "Agent": ["Base", "Base"],
            "Combination": ["Run1", "Run2"],
            "AnnReturn": [0.045, 0.035],
            "AnnVol": [0.11, 0.115],
        }
    )

    _, metric_diff = build_run_diff({}, {}, current_summary, previous_summary)

    assert {"Agent", "Combination"} <= set(metric_diff.columns)
    run1 = metric_diff[metric_diff["Combination"] == "Run1"]
    run2 = metric_diff[metric_diff["Combination"] == "Run2"]
    assert not run1.empty
    assert not run2.empty


def test_build_run_diff_handles_object_numeric_columns() -> None:
    current_summary = pd.DataFrame(
        {
            "Agent": ["Base", "Alt"],
            "AnnReturn": [0.05, 0.03],
            "TE": [None, 0.2],
        }
    )
    previous_summary = pd.DataFrame(
        {
            "Agent": ["Base", "Alt"],
            "AnnReturn": [0.04, 0.02],
            "TE": [None, 0.25],
        }
    )

    _, metric_diff = build_run_diff({}, {}, current_summary, previous_summary)

    te_rows = metric_diff[metric_diff["Metric"] == "TE"]
    assert not te_rows.empty
    alt_te = te_rows[te_rows["Agent"] == "Alt"].iloc[0]
    assert alt_te["Delta"] == pytest.approx(-0.05)


def test_build_run_diff_parses_percent_strings() -> None:
    current_summary = pd.DataFrame(
        {
            "Agent": ["Base"],
            "AnnReturn": ["5.0%"],
            "AnnVol": ["10%"],
        }
    )
    previous_summary = pd.DataFrame(
        {
            "Agent": ["Base"],
            "AnnReturn": ["4.0%"],
            "AnnVol": ["12%"],
        }
    )

    _, metric_diff = build_run_diff({}, {}, current_summary, previous_summary)

    ann_return = metric_diff[metric_diff["Metric"] == "AnnReturn"].iloc[0]
    ann_vol = metric_diff[metric_diff["Metric"] == "AnnVol"].iloc[0]
    assert ann_return["Delta"] == pytest.approx(0.01)
    assert ann_vol["Delta"] == pytest.approx(-0.02)
