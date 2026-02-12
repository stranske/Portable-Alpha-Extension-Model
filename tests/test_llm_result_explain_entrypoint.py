"""Contract tests for pa_core.llm.result_explain entrypoint."""

from __future__ import annotations

import pandas as pd
import pytest

from pa_core.contracts import (
    SUMMARY_BREACH_PROB_COLUMN,
    SUMMARY_CVAR_COLUMN,
    SUMMARY_TE_COLUMN,
    SUMMARY_TRACKING_ERROR_LEGACY_COLUMN,
)
from pa_core.llm.result_explain import explain_results_details


def test_explain_results_details_returns_expected_tuple_contract():
    df = pd.DataFrame(
        {
            "monthly_TE": [0.02, 0.03],
            "monthly_CVaR": [-0.05, -0.04],
            "Agent": ["A", "B"],
        }
    )
    manifest = {"seed": 42, "cli_args": {"output": "results.xlsx", "n_sims": 1000}}

    text, trace_url, payload = explain_results_details(df, manifest)

    assert isinstance(text, str)
    assert len(text) > 0
    assert trace_url is None
    assert isinstance(payload, dict)
    assert payload["rows"] == 2
    assert payload["columns"] == ["monthly_TE", "monthly_CVaR", "Agent"]
    assert payload["manifest_highlights"]["seed"] == 42
    assert payload["manifest_highlights"]["cli_output"] == "results.xlsx"
    assert payload["manifest_highlights"]["cli_n_sims"] == 1000
    assert payload["metric_catalog"]["monthly_TE"] == pytest.approx(0.025)
    assert payload["metric_catalog"]["monthly_CVaR"] == pytest.approx(-0.045)


def test_explain_results_details_accepts_missing_manifest():
    df = pd.DataFrame({"metric": [1.0, 2.0, 3.0]})

    text, trace_url, payload = explain_results_details(df, None)

    assert isinstance(text, str)
    assert trace_url is None
    assert payload["manifest_highlights"] == {}
    assert payload["mean_stats"]["metric"] == pytest.approx(2.0)
    assert payload["metric_catalog"] == {}


def test_explain_results_details_rejects_non_dataframe():
    with pytest.raises(TypeError, match="details_df must be a pandas DataFrame"):
        explain_results_details({"not": "a dataframe"})


def test_explain_results_details_extracts_summary_metric_catalog():
    summary_df = pd.DataFrame(
        {
            SUMMARY_TE_COLUMN: [0.01, 0.03, 0.02],
            SUMMARY_TRACKING_ERROR_LEGACY_COLUMN: [0.02, 0.04, 0.06],
            SUMMARY_CVAR_COLUMN: [-0.05, -0.07, -0.06],
            SUMMARY_BREACH_PROB_COLUMN: [0.10, 0.20, 0.15],
            "label": ["A", "B", "C"],
        }
    )

    _, _, payload = explain_results_details(summary_df, manifest=None)

    assert payload["metric_catalog"] == {
        SUMMARY_TE_COLUMN: pytest.approx(0.02),
        SUMMARY_TRACKING_ERROR_LEGACY_COLUMN: pytest.approx(0.04),
        SUMMARY_CVAR_COLUMN: pytest.approx(-0.06),
        SUMMARY_BREACH_PROB_COLUMN: pytest.approx(0.15),
    }


def test_explain_results_details_metric_catalog_handles_missing_and_null_columns():
    summary_df = pd.DataFrame(
        {
            SUMMARY_TE_COLUMN: [None, None],
            SUMMARY_BREACH_PROB_COLUMN: [0.05, None],
            "non_numeric": ["x", "y"],
        }
    )

    _, _, payload = explain_results_details(summary_df, manifest=None)

    assert SUMMARY_TE_COLUMN not in payload["metric_catalog"]
    assert SUMMARY_TRACKING_ERROR_LEGACY_COLUMN not in payload["metric_catalog"]
    assert SUMMARY_CVAR_COLUMN not in payload["metric_catalog"]
    assert payload["metric_catalog"][SUMMARY_BREACH_PROB_COLUMN] == pytest.approx(0.05)
