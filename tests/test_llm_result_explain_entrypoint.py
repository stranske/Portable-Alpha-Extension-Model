"""Contract tests for pa_core.llm.result_explain entrypoint."""

from __future__ import annotations

import pandas as pd
import pytest

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
    assert payload["analysis_output"]["rows"] == 2
    assert payload["analysis_output"]["columns"] == ["monthly_TE", "monthly_CVaR", "Agent"]
    assert payload["analysis_output"]["manifest_highlights"]["seed"] == 42
    assert payload["analysis_output"]["manifest_highlights"]["cli_output"] == "results.xlsx"
    assert payload["analysis_output"]["manifest_highlights"]["cli_n_sims"] == 1000
    assert "LLM configuration is required" in text


def test_explain_results_details_accepts_missing_manifest():
    df = pd.DataFrame({"metric": [1.0, 2.0, 3.0]})

    text, trace_url, payload = explain_results_details(df, None)

    assert isinstance(text, str)
    assert trace_url is None
    assert payload["analysis_output"]["manifest_highlights"] == {}
    assert payload["analysis_output"]["basic_statistics"]["metric"]["mean"] == pytest.approx(2.0)


def test_explain_results_details_rejects_non_dataframe():
    with pytest.raises(TypeError, match="details_df must be a pandas DataFrame"):
        explain_results_details({"not": "a dataframe"})
