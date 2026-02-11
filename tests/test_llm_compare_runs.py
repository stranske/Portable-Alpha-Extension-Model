"""Tests for pa_core.llm.compare_runs helpers."""

from __future__ import annotations

import json

from pa_core.llm.compare_runs import load_prior_manifest


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
