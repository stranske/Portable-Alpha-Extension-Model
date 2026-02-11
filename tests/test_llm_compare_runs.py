"""Tests for pa_core.llm.compare_runs helpers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pa_core.llm.compare_runs import (
    build_metric_catalog,
    compare_runs,
    format_config_diff,
    load_prior_manifest,
    load_prior_summary,
)


def test_load_prior_manifest_reads_previous_run_file(tmp_path):
    prev_manifest = {"seed": 99, "cli_args": {"output": "old.xlsx"}}
    prev_manifest_path = tmp_path / "previous-manifest.json"
    prev_manifest_path.write_text(json.dumps(prev_manifest))

    loaded, path = load_prior_manifest({"previous_run": str(prev_manifest_path)})

    assert loaded == prev_manifest
    assert path == prev_manifest_path


def test_load_prior_manifest_returns_none_when_previous_run_missing():
    loaded, path = load_prior_manifest({"seed": 7})

    assert loaded is None
    assert path is None


def test_load_prior_manifest_returns_path_when_file_missing(tmp_path):
    missing_path = tmp_path / "does-not-exist.json"

    loaded, path = load_prior_manifest({"previous_run": str(missing_path)})

    assert loaded is None
    assert path == missing_path


def test_load_prior_summary_reads_previous_output(tmp_path):
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.02], "monthly_CVaR": [-0.03]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest = {"cli_args": {"output": str(prior_output)}}

    loaded, path = load_prior_summary(prior_manifest)

    assert isinstance(loaded, pd.DataFrame)
    assert path == prior_output
    assert "monthly_TE" in loaded.columns


def test_format_config_diff_includes_seed_cli_and_wizard_changes():
    current_manifest = {
        "seed": 11,
        "cli_args": {"capital": 1000000, "distribution": "gaussian"},
        "wizard_config": {"risk": {"cvar_limit": 0.07}},
    }
    prior_manifest = {
        "seed": 7,
        "cli_args": {"capital": 750000, "distribution": "student_t"},
        "wizard_config": {"risk": {"cvar_limit": 0.09}},
    }

    text = format_config_diff(current_manifest, prior_manifest)

    assert "seed" in text
    assert "cli_args.capital" in text
    assert "wizard_config.risk.cvar_limit" in text


def test_build_metric_catalog_extracts_supported_metrics():
    df = pd.DataFrame(
        {
            "monthly_TE": [0.02, 0.03],
            "TrackingErr": [0.01, 0.02],
            "monthly_CVaR": [-0.04, -0.06],
            "monthly_BreachProb": [0.1, 0.2],
            "other": ["a", "b"],
        }
    )

    catalog = build_metric_catalog(df)

    assert catalog["monthly_TE"] == 0.025
    assert catalog["TrackingErr"] == 0.015
    assert catalog["monthly_CVaR"] == -0.05
    assert catalog["monthly_BreachProb"] == pytest.approx(0.15)


def test_compare_runs_returns_text_trace_and_payload(monkeypatch, tmp_path):
    current_summary = pd.DataFrame({"monthly_TE": [0.03], "monthly_CVaR": [-0.05]})
    prior_output = tmp_path / "prior.xlsx"
    pd.DataFrame({"monthly_TE": [0.01], "monthly_CVaR": [-0.04]}).to_excel(
        prior_output, sheet_name="Summary", index=False
    )
    prior_manifest_path = tmp_path / "prior_manifest.json"
    prior_manifest_path.write_text(
        json.dumps({"cli_args": {"output": str(prior_output)}, "seed": 1})
    )

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")

    text, trace_url, payload = compare_runs(
        current_summary=current_summary,
        current_manifest={"previous_run": str(prior_manifest_path), "seed": 2},
        questions="What changed?",
        provider_name="openai",
        model_name="gpt-4o-mini",
    )

    assert isinstance(text, str)
    assert "comparison" in text.lower() or len(text) > 0
    assert trace_url is not None
    assert payload.config_diff
    assert payload.prior_manifest_path == str(prior_manifest_path)
