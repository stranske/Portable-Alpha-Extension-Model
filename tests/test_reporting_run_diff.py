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

    assert set(metric_diff["Metric"]) == {"AnnReturn", "AnnVol"}
    ann_return = metric_diff[metric_diff["Metric"] == "AnnReturn"].iloc[0]
    assert ann_return["Delta"] == pytest.approx(0.01)


def test_build_run_diff_handles_empty_summaries() -> None:
    empty = pd.DataFrame()
    cfg_diff, metric_diff = build_run_diff(None, None, empty, empty)

    assert cfg_diff.empty
    assert metric_diff.empty
