"""Tests for the unified run-record envelope (run.json) and captured warnings.

Covers issue #1834: every run must be representable as a single JSON object that
also answers "did this run warn about anything?". The frequency-mismatch test is
failing-first on ``main`` (no ``warnings`` field / no ``run.json`` exists today).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

yaml: Any = pytest.importorskip("yaml")

# Match the lightweight stubs used by tests/test_manifest.py so importing the CLI
# does not pull in streamlit / python-pptx.
sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
pptx_mod = types.ModuleType("pptx")
pptx_util = types.ModuleType("pptx.util")
pptx_mod.Presentation = object  # type: ignore[attr-defined]
pptx_util.Inches = lambda x: x  # type: ignore[attr-defined]
pptx_mod.util = pptx_util  # type: ignore[attr-defined]
sys.modules.setdefault("pptx", pptx_mod)
sys.modules.setdefault("pptx.util", pptx_util)

from pa_core.cli import main  # noqa: E402
from pa_core.contracts import (  # noqa: E402
    MANIFEST_OPTIONAL_FIELDS,
    RUN_RECORD_WARNING_FIELDS,
)

DATA_INDEX = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"


def _run_cli(tmp_path: Path, *extra: str) -> Path:
    cfg = {"N_SIMULATIONS": 1, "N_MONTHS": 1, "financing_mode": "broadcast"}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out_file = tmp_path / "out.xlsx"
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(DATA_INDEX),
            "--output",
            str(out_file),
            "--seed",
            "123",
            *extra,
        ]
    )
    return out_file


def test_frequency_mismatch_warning_serialized(tmp_path: Path) -> None:
    # Declaring a non-monthly --index-frequency overrides detection and surfaces a
    # genuine "Index frequency mismatch" UserWarning (non-strict) at a real warn
    # site in pa_core.data.loaders, which the collector must serialize.
    out_file = _run_cli(tmp_path, "--index-frequency", "daily")

    run_record_path = out_file.with_name("run.json")
    assert run_record_path.exists(), "run.json envelope was not written"
    record = json.loads(run_record_path.read_text())

    assert "warnings" in record and isinstance(record["warnings"], list)
    assert "cost" in record
    messages = [str(w.get("message", "")) for w in record["warnings"]]
    assert any("frequency mismatch" in m.lower() for m in messages), (
        f"expected a captured frequency-mismatch warning, got: {messages}"
    )
    # Every captured warning is normalized to the four-key shape.
    for warning in record["warnings"]:
        assert set(RUN_RECORD_WARNING_FIELDS).issubset(warning.keys())

    # run.json references the sibling artifacts.
    assert record["manifest_path"] and Path(record["manifest_path"]).name == "manifest.json"


def test_run_record_cost_stub_present(tmp_path: Path) -> None:
    out_file = _run_cli(tmp_path)
    record = json.loads(out_file.with_name("run.json").read_text())
    cost = record["cost"]
    assert cost is not None
    assert "latency_seconds" in cost and isinstance(cost["latency_seconds"], (int, float))
    # Dollar cost is a deliberate stub for the local numpy backend.
    assert cost["dollars"] is None


def test_run_record_bundle_path_points_to_bundle_json(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    out_file = _run_cli(tmp_path, "--bundle", str(bundle_dir))

    record = json.loads(out_file.with_name("run.json").read_text())

    assert record["bundle_path"] == str(bundle_dir / "bundle.json")
    assert Path(record["bundle_path"]).is_file()


def test_manifest_carries_optional_warnings_and_cost(tmp_path: Path) -> None:
    out_file = _run_cli(tmp_path, "--index-frequency", "daily")
    manifest = json.loads(out_file.with_name("manifest.json").read_text())
    # Additive optional fields are present on freshly-written manifests.
    assert "warnings" in MANIFEST_OPTIONAL_FIELDS and "cost" in MANIFEST_OPTIONAL_FIELDS
    assert "warnings" in manifest and isinstance(manifest["warnings"], list)
    assert "cost" in manifest
