"""Regression coverage for the #2375 comparison and CLI error paths."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pa_core
import pandas as pd
import pytest
import yaml

from pa_core.cli import main

from pa_core.llm.compare_runs import (
    _largest_metric_delta,
    build_metric_catalog,
    compare_runs,
    load_prior_manifest,
    load_prior_summary,
)


def test_compare_run_helpers_reject_unusable_artifacts(tmp_path) -> None:
    """Unreadable or structurally invalid artifacts must not look usable."""
    non_mapping_path = tmp_path / "not-a-manifest.json"
    non_mapping_path.write_text(json.dumps(["not", "a", "mapping"]))

    assert load_prior_manifest(None) == (None, None)
    assert load_prior_manifest({"previous_run": str(non_mapping_path)}) == (
        None,
        non_mapping_path,
    )
    assert load_prior_summary(None) == (None, None)
    assert load_prior_summary({"cli_args": "not-a-mapping"}) == (None, None)


def test_compare_runs_explains_missing_prior_summary(tmp_path) -> None:
    """A valid prior manifest without a workbook is a distinct user error."""
    prior_manifest_path = tmp_path / "prior-manifest.json"
    prior_manifest_path.write_text(json.dumps({"cli_args": {"output": "missing.xlsx"}}))

    with pytest.raises(ValueError, match="No readable prior summary found"):
        compare_runs(
            current_summary=pd.DataFrame({"monthly_TE": [0.01]}),
            current_manifest={"previous_run": str(prior_manifest_path)},
            questions="What changed?",
        )


def test_compare_runs_rejects_missing_prior_manifest() -> None:
    """A request without a previous manifest cannot produce a comparison."""
    with pytest.raises(ValueError, match="No readable prior manifest"):
        compare_runs(
            current_summary=pd.DataFrame({"monthly_TE": [0.01]}),
            current_manifest=None,
            questions="What changed?",
        )


def test_compare_runs_survives_fleet_event_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telemetry failure must not discard an otherwise valid comparison."""
    prior_summary = pd.DataFrame({"monthly_TE": [0.01]})
    monkeypatch.setattr(
        "pa_core.llm.compare_runs.load_prior_manifest",
        lambda _manifest: ({"seed": 1, "cli_args": {"output": "prior.xlsx"}}, Path("prior.json")),
    )
    monkeypatch.setattr(
        "pa_core.llm.compare_runs.load_prior_summary",
        lambda _manifest, *, manifest_path: (prior_summary, Path("prior.xlsx")),
    )
    monkeypatch.setattr(
        "pa_core.llm.compare_runs._invoke_comparison_llm",
        lambda *_args, **_kwargs: "comparison complete",
    )
    monkeypatch.setattr(
        "pa_core.llm.compare_runs.record_fleet_event",
        lambda _event: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
    )

    text, _, payload = compare_runs(
        current_summary=pd.DataFrame({"monthly_TE": [0.02]}),
        current_manifest={"seed": 2},
        questions="What changed?",
    )

    assert text == "comparison complete"
    assert payload.status == "success"


def test_metric_catalog_rejects_non_dataframe_and_empty_delta() -> None:
    """Comparison input validation cannot silently turn invalid inputs into metrics."""
    with pytest.raises(TypeError, match="pandas DataFrame"):
        build_metric_catalog("not-a-dataframe")

    assert _largest_metric_delta({"current": 0.1}, {"prior": 0.2}) is None


def _write_single_run_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "N_SIMULATIONS": 1,
                "N_MONTHS": 1,
                "financing_mode": "broadcast",
                "analysis_mode": "single_with_sensitivity",
            }
        )
    )
    index = Path(__file__).resolve().parents[1] / "data/sp500tr_fred_divyield.csv"
    return config, index, tmp_path / "result.xlsx"


def _install_fake_viz(monkeypatch: pytest.MonkeyPatch) -> None:
    figure = object()
    viz = types.ModuleType("pa_core.viz")
    viz.risk_return = types.SimpleNamespace(make=lambda _df: figure)
    viz.sharpe_ladder = types.SimpleNamespace(make=lambda _df: figure)
    monkeypatch.setitem(sys.modules, "pa_core.viz", viz)
    monkeypatch.setattr(pa_core, "viz", viz)
    monkeypatch.setattr("pa_core.cli.print_enhanced_summary", lambda *_args, **_kwargs: None)


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        (RuntimeError, "Export packet failed: packet unavailable"),
        (ImportError, "Export packet failed due to missing dependency: packet unavailable"),
        (ValueError, "Export packet failed due to data or configuration issue: packet unavailable"),
        (OSError, "Export packet failed due to file system issue: packet unavailable"),
    ],
)
def test_single_run_packet_failure_reports_the_failure_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
    expected: str,
) -> None:
    """Packet failures must be specific and must finish the run cleanly."""
    monkeypatch.chdir(tmp_path)
    _install_fake_viz(monkeypatch)
    config, index, output = _write_single_run_config(tmp_path)

    def fail_packet(**_kwargs: object) -> tuple[Path, Path]:
        raise error_type("packet unavailable")

    monkeypatch.setattr("pa_core.reporting.export_packet.create_export_packet", fail_packet)

    main(["--config", str(config), "--index", str(index), "--output", str(output), "--packet"])

    assert f"❌ {expected}" in capsys.readouterr().out


def test_single_run_gif_conversion_error_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid return arrays must not proceed to an animation export."""
    monkeypatch.chdir(tmp_path)
    _install_fake_viz(monkeypatch)
    config, index, output = _write_single_run_config(tmp_path)
    monkeypatch.setattr(
        "pa_core.viz.utils.safe_to_numpy",
        lambda _value: (_ for _ in ()).throw(ValueError("not numeric")),
    )

    main(["--config", str(config), "--index", str(index), "--output", str(output), "--gif"])

    assert "❌ GIF export failed: Data conversion error - not numeric" in capsys.readouterr().out


def test_single_run_dashboard_missing_file_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dashboard path is checked before any launcher is invoked."""
    monkeypatch.chdir(tmp_path)
    _install_fake_viz(monkeypatch)
    config, index, output = _write_single_run_config(tmp_path)

    main(["--config", str(config), "--index", str(index), "--output", str(output), "--dashboard"])

    assert "❌ Dashboard launch failed: Dashboard file not found" in capsys.readouterr().out


def test_single_run_bundle_contains_the_completed_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The opt-in bundle records outputs after the normal run has completed."""
    monkeypatch.chdir(tmp_path)
    config, index, output = _write_single_run_config(tmp_path)
    bundle_dir = tmp_path / "bundle"
    monkeypatch.setattr("pa_core.cli.print_enhanced_summary", lambda *_args, **_kwargs: None)

    main(
        [
            "--config",
            str(config),
            "--index",
            str(index),
            "--output",
            str(output),
            "--bundle",
            str(bundle_dir),
        ]
    )

    assert (bundle_dir / "bundle.json").is_file()
