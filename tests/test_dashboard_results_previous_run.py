from __future__ import annotations

import json
import runpy
from pathlib import Path


def _load_results_module() -> dict[str, object]:
    return runpy.run_path(str(Path("dashboard/pages/4_Results.py")))


def test_check_previous_run_availability_missing_field() -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    result = check({"seed": 7})

    assert result.available is False
    assert result.prior_manifest_path is None
    assert "missing artifact" in str(result.message).lower()
    assert "previous_run" in str(result.message)


def test_check_previous_run_availability_missing_file_path(tmp_path) -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    missing = tmp_path / "missing_manifest.json"
    result = check({"previous_run": str(missing)})

    assert result.available is False
    assert result.prior_manifest_path == missing
    assert str(missing) in str(result.message)
    assert "expected file path" in str(result.message).lower()


def test_check_previous_run_availability_unreadable_json(tmp_path) -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    invalid_manifest = tmp_path / "invalid_manifest.json"
    invalid_manifest.write_text("{broken-json")
    result = check({"previous_run": str(invalid_manifest)})

    assert result.available is False
    assert result.prior_manifest_path == invalid_manifest
    assert "unreadable artifact" in str(result.message).lower()
    assert str(invalid_manifest) in str(result.message)


def test_check_previous_run_availability_readable_manifest(tmp_path) -> None:
    module = _load_results_module()
    check = module["_check_previous_run_availability"]
    prior_manifest = tmp_path / "prior_manifest.json"
    prior_manifest.write_text(json.dumps({"seed": 42}))
    result = check({"previous_run": str(prior_manifest)})

    assert result.available is True
    assert result.prior_manifest_path == prior_manifest
    assert result.message is None
