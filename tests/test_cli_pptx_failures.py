"""Single-run PowerPoint failures keep saved results and report the actual failure class."""

import json
import logging
import sys
import types
from pathlib import Path

import pytest
import yaml

import pa_core
from pa_core.cli import main


@pytest.mark.parametrize(
    "error_type, message, diagnostic, guidance, log_diagnostic",
    [
        (
            ImportError,
            "kaleido unavailable",
            "PPTX export failed: Kaleido or Chrome/Chromium required",
            "Install with: pip install kaleido (preferred) or sudo apt-get install chromium-browser",
            "PPTX export failed due to missing dependency",
        ),
        (
            ModuleNotFoundError,
            "plot helper unavailable",
            "PPTX export failed due to missing dependency: plot helper unavailable",
            None,
            "PPTX export failed due to missing module",
        ),
        *[
            (
                error,
                "destination unavailable",
                "PPTX export failed: Cannot write file - destination unavailable",
                "Check file permissions and available disk space",
                "PPTX export failed due to file system issue",
            )
            for error in (OSError, PermissionError)
        ],
        *[
            (
                error,
                "invalid chart data",
                "PPTX export failed: Invalid data - invalid chart data",
                "Check your visualization data and parameters",
                "PPTX export failed due to data issue",
            )
            for error in (ValueError, TypeError)
        ],
        (
            RuntimeError,
            "chromium renderer cancelled",
            "PPTX export failed: Kaleido or Chrome/Chromium required",
            "Install with: pip install kaleido (preferred) or sudo apt-get install chromium-browser",
            "PPTX export failed",
        ),
        (
            ModuleNotFoundError,
            "python-pptx unavailable",
            "PPTX export failed: python-pptx required",
            "Install with: pip install python-pptx",
            "PPTX export failed due to missing python-pptx",
        ),
    ],
    ids=[
        "renderer",
        "module",
        "io",
        "permission",
        "value",
        "type",
        "runtime-renderer",
        "python-pptx",
    ],
)
def test_single_run_pptx_failure_preserves_results(
    monkeypatch,
    tmp_path,
    capsys,
    caplog,
    error_type,
    message,
    diagnostic,
    guidance,
    log_diagnostic,
):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger="pa_core.cli")
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
    output = tmp_path / "result.xlsx"
    manifest_path = tmp_path / "manifest.json"
    figure = object()
    tornado = object()
    calls = []
    before = []

    def fail_pptx(figures, filename, *, alt_texts):
        calls.append((figures, filename, alt_texts))
        before.append((output.read_bytes(), json.loads(manifest_path.read_text())))
        raise error_type(message)

    viz = types.ModuleType("pa_core.viz")
    viz.risk_return = types.SimpleNamespace(make=lambda _df: figure)
    viz.sharpe_ladder = types.SimpleNamespace(make=lambda _df: figure)
    monkeypatch.setitem(sys.modules, "pa_core.viz", viz)
    monkeypatch.setattr(pa_core, "viz", viz)
    viz.pptx_export = types.SimpleNamespace(save=fail_pptx)
    viz.tornado = types.SimpleNamespace(
        series_from_sensitivity=lambda _df: "sensitivity-series",
        make=lambda series, *, title=None: tornado,
    )
    monkeypatch.setattr("pa_core.cli.print_enhanced_summary", lambda *_a, **_k: None)
    index = Path(__file__).resolve().parents[1] / "data/sp500tr_fred_divyield.csv"
    main(
        [
            "--config",
            str(config),
            "--index",
            str(index),
            "--output",
            str(output),
            "--seed",
            "123",
            "--pptx",
            "--alt-text",
            "Portfolio summary",
            "--sensitivity",
        ]
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines() + captured.err.splitlines()
    assert calls == [([figure, tornado], "plots/summary.pptx", ["Portfolio summary"])]
    assert [line for line in lines if "PPTX export failed" in line] == [f"❌ {diagnostic}"]
    assert [line for line in lines if line.startswith("💡 ")] == (
        [] if guidance is None else [f"💡 {guidance}"]
    )
    assert sum(record.levelno == logging.ERROR for record in caplog.records) == 1
    assert [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if "PPTX export" in record.getMessage()
    ] == [(logging.ERROR, f"{log_diagnostic}: {message}")]
    assert output.read_bytes() == before[0][0]
    assert not (tmp_path / "plots/summary.pptx").exists()
    final_manifest = json.loads(manifest_path.read_text())
    assert final_manifest["config"] == before[0][1]["config"]
    assert final_manifest["seed"] == 123
    assert any(log_diagnostic in warning["message"] for warning in final_manifest["warnings"])
