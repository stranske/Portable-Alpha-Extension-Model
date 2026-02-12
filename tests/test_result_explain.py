"""Unit tests for metric extraction behavior in pa_core.llm.result_explain."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pa_core.llm.result_explain import explain_results_details


@pytest.fixture
def baseline_summary_df() -> pd.DataFrame:
    """Summary dataframe with all expected metric columns populated."""
    return pd.DataFrame(
        {
            "monthly_TE": [0.02, 0.03, 0.01],
            "monthly_CVaR": [-0.05, -0.04, -0.06],
            "Sharpe": [1.1, 1.2, 1.3],
            "Regime": ["base", "bull", "bear"],
            "Agent": ["A", "B", "C"],
        }
    )


@pytest.fixture
def summary_df_missing_monthly_cvar(baseline_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Summary dataframe variant with one required metric column removed."""
    return baseline_summary_df.drop(columns=["monthly_CVaR"])


@pytest.fixture
def summary_df_with_null_nan_empty() -> pd.DataFrame:
    """Summary dataframe variant with None/NaN/empty-string metric values."""
    return pd.DataFrame(
        {
            "monthly_TE": [0.02, None, np.nan],
            "monthly_CVaR": [-0.05, np.nan, -0.01],
            "Sharpe": [None, np.nan, None],
            "Regime": ["base", "", None],
            "Agent": ["A", "", None],
        }
    )


def test_explain_results_details_extracts_expected_metrics_baseline(
    baseline_summary_df: pd.DataFrame,
) -> None:
    text, trace_url, payload = explain_results_details(baseline_summary_df)

    assert text == "Result explanation is ready. Processed 3 rows across 5 columns."
    assert trace_url is None
    assert payload["rows"] == 3
    assert payload["columns"] == [
        "monthly_TE",
        "monthly_CVaR",
        "Sharpe",
        "Regime",
        "Agent",
    ]
    assert payload["mean_stats"]["monthly_TE"] == pytest.approx(0.02)
    assert payload["mean_stats"]["monthly_CVaR"] == pytest.approx(-0.05)


def test_explain_results_details_handles_missing_metric_column(
    summary_df_missing_monthly_cvar: pd.DataFrame,
) -> None:
    text, trace_url, payload = explain_results_details(summary_df_missing_monthly_cvar)

    assert text == "Result explanation is ready. Processed 3 rows across 4 columns."
    assert trace_url is None
    assert payload["columns"] == ["monthly_TE", "Sharpe", "Regime", "Agent"]
    assert payload["mean_stats"] == {
        "monthly_TE": pytest.approx(0.02),
        "Sharpe": pytest.approx(1.2),
    }
    assert "monthly_CVaR" not in payload["mean_stats"]


def test_explain_results_details_handles_null_nan_empty_metric_values(
    summary_df_with_null_nan_empty: pd.DataFrame,
) -> None:
    text, trace_url, payload = explain_results_details(summary_df_with_null_nan_empty)

    assert text == "Result explanation is ready. Processed 3 rows across 5 columns."
    assert trace_url is None
    assert payload["rows"] == 3
    assert payload["columns"] == [
        "monthly_TE",
        "monthly_CVaR",
        "Sharpe",
        "Regime",
        "Agent",
    ]
    assert payload["mean_stats"]["monthly_TE"] == pytest.approx(0.02)
    assert payload["mean_stats"]["monthly_CVaR"] == pytest.approx(-0.03)
    assert math.isnan(payload["mean_stats"]["Sharpe"])
