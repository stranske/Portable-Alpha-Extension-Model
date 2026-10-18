"""Optional single-run PDF failures preserve results and truthful diagnostics."""

import json
import logging
import sys
import types
from pathlib import Path

import pandas as pd
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
            "PDF export failed: Kaleido or Chrome/Chromium required",
            "Install with: pip install kaleido (preferred) or sudo apt-get install chromium-browser",
            "PDF export failed due to missing dependency",
        ),
        (
            ModuleNotFoundError,
            "pdf helper unavailable",
            "PDF export failed due to missing dependency: pdf helper unavailable",
            None,
            "PDF export failed due to missing module",
        ),
        *[
            (
                error,
                "destination unavailable",
                "PDF export failed: Cannot write file - destination unavailable",
                "Check file permissions and available disk space",
                "PDF export failed due to file system issue",
            )
            for error in (OSError, PermissionError)
        ],
        *[
            (
                error,
                "invalid chart data",
                "PDF export failed: Invalid data - invalid chart data",
                "Check your visualization data and parameters",
                "PDF export failed due to data issue",
            )
            for error in (ValueError, TypeError)
        ],
    ],
    ids=["renderer", "module", "io", "permission", "value", "type"],
)
def test_single_run_pdf_failure_preserves_results(
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
    calls = []
    before = []
    figure = object()

    def fail_pdf(fig, filename):
        calls.append((fig, filename))
        before.append((output.read_bytes(), json.loads(manifest_path.read_text())))
        raise error_type(message)

    viz = types.ModuleType("pa_core.viz")
    viz.risk_return = types.SimpleNamespace(make=lambda _df: figure)
    viz.sharpe_ladder = types.SimpleNamespace(make=lambda _df: figure)
    viz.pdf_export = types.SimpleNamespace(save=fail_pdf)
    monkeypatch.setitem(sys.modules, "pa_core.viz", viz)
    monkeypatch.setattr(pa_core, "viz", viz)
    monkeypatch.setattr("pa_core.cli.print_enhanced_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "pa_core.sim.sensitivity.one_factor_deltas", lambda *_a, **_k: pd.DataFrame()
    )
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
            "--pdf",
            "--sensitivity",
        ]
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines() + captured.err.splitlines()
    assert calls == [(figure, "plots/summary.pdf")]
    assert [line for line in lines if "PDF export failed" in line] == [f"❌ {diagnostic}"]
    assert [line for line in lines if line.startswith("💡 ")] == (
        [] if guidance is None else [f"💡 {guidance}"]
    )
    assert [
        (r.levelno, r.getMessage()) for r in caplog.records if "PDF export" in r.getMessage()
    ] == [(logging.ERROR, f"{log_diagnostic}: {message}")]
    assert output.read_bytes() == before[0][0]
    assert not (tmp_path / "plots/summary.pdf").exists()
    final_manifest = json.loads(manifest_path.read_text())
    assert final_manifest["config"] == before[0][1]["config"]
    assert final_manifest["seed"] == 123
    assert any(log_diagnostic in warning["message"] for warning in final_manifest["warnings"])
