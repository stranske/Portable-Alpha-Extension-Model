"""Sweep packet failures stay diagnostic and do not announce nonexistent exports."""

import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pa_core.cli import main


@pytest.mark.parametrize(
    "error_type, diagnostic, guidance",
    [
        (RuntimeError, "Export packet failed: export unavailable", None),
        (
            ImportError,
            "Export packet failed due to missing dependency",
            "Install required packages",
        ),
        (
            ModuleNotFoundError,
            "Export packet failed due to missing dependency",
            "Install required packages",
        ),
        (
            ValueError,
            "Export packet failed due to data issue",
            "Check your configuration and data inputs",
        ),
        (
            TypeError,
            "Export packet failed due to data issue",
            "Check your configuration and data inputs",
        ),
        (
            KeyError,
            "Export packet failed due to data issue",
            "Check your configuration and data inputs",
        ),
    ],
    ids=["runtime", "import", "module", "value", "type", "key"],
)
def test_sweep_packet_export_failure_is_truthful(
    monkeypatch, tmp_path, capsys, error_type, diagnostic, guidance
):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "N_SIMULATIONS": 1,
                "N_MONTHS": 1,
                "financing_mode": "broadcast",
                "analysis_mode": "returns",
            }
        )
    )
    summary = pd.DataFrame(
        {
            "Agent": ["Base"],
            "terminal_AnnReturn": [0.06],
            "monthly_AnnVol": [0.11],
            "terminal_ShortfallProb": [0.08],
        }
    )
    monkeypatch.setattr(
        "pa_core.sweep.run_parameter_sweep",
        lambda *_a, **_k: [{"summary": summary, "combination_id": 1}],
    )

    def export_sweep(_results, filename="Sweep.xlsx", **_kwargs):
        Path(filename).write_text("sweep output")

    monkeypatch.setattr("pa_core.reporting.sweep_excel.export_sweep_results", export_sweep)
    calls = []

    def fail_packet(**kwargs):
        calls.append(kwargs)
        raise error_type("export unavailable")

    monkeypatch.setattr("pa_core.reporting.export_packet.create_export_packet", fail_packet)
    viz = types.ModuleType("pa_core.viz")
    viz.risk_return = types.SimpleNamespace(make=lambda _df: object())
    viz.sharpe_ladder = types.SimpleNamespace(make=lambda _df: object())
    viz.theme = types.SimpleNamespace(DEFAULT_SHORTFALL_PROB=0.1)
    monkeypatch.setitem(sys.modules, "pa_core.viz", viz)
    index = Path(__file__).resolve().parents[1] / "data/sp500tr_fred_divyield.csv"
    output = tmp_path / "sweep.xlsx"
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
            "--packet",
        ]
    )
    stdout = capsys.readouterr().out
    assert len(calls) == 1
    pd.testing.assert_frame_equal(calls[0]["summary_df"][list(summary.columns)], summary)
    assert diagnostic in stdout
    assert "export unavailable" in stdout
    if guidance:
        assert guidance in stdout
    else:
        assert "Install required packages" not in stdout
        assert "Check your configuration and data inputs" not in stdout
    assert "Parameter sweep export packet created" not in stdout
    assert output.read_text() == "sweep output"
    assert (tmp_path / "manifest.json").is_file()
