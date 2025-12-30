import json
import sys
import types
from pathlib import Path

import pandas as pd
import yaml

from pa_core.cli import main


def test_sweep_packet_passes_prev_diff(monkeypatch, tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "N_SIMULATIONS": 1,
                "N_MONTHS": 1,
                "analysis_mode": "returns",
            }
        )
    )

    prev_summary = pd.DataFrame(
        {
            "Agent": ["Base"],
            "AnnReturn": [0.05],
            "AnnVol": [0.12],
            "ShortfallProb": [0.1],
        }
    )
    prev_output = tmp_path / "prev.xlsx"
    prev_summary.to_excel(prev_output, sheet_name="Summary", index=False)

    prev_manifest = {
        "cli_args": {"output": str(prev_output)},
        "config": {"N_SIMULATIONS": 1},
    }
    prev_manifest_path = tmp_path / "manifest.json"
    prev_manifest_path.write_text(json.dumps(prev_manifest))

    def _stub_run_parameter_sweep(*_args, **_kwargs):
        summary = pd.DataFrame(
            {
                "Agent": ["Base"],
                "AnnReturn": [0.06],
                "AnnVol": [0.11],
                "ShortfallProb": [0.08],
            }
        )
        return [{"summary": summary, "combination_id": 1}]

    def _stub_export_sweep_results(_results, filename="Sweep.xlsx"):
        Path(filename).write_text("stub")

    captured: dict[str, object] = {}

    def _stub_create_export_packet(**kwargs):
        captured.update(kwargs)
        return (str(tmp_path / "out.pptx"), str(tmp_path / "out.xlsx"))

    viz_stub = types.ModuleType("pa_core.viz")
    viz_stub.risk_return = types.SimpleNamespace(make=lambda _df: object())
    viz_stub.sharpe_ladder = types.SimpleNamespace(make=lambda _df: object())
    monkeypatch.setitem(sys.modules, "pa_core.viz", viz_stub)
    monkeypatch.setattr("pa_core.sweep.run_parameter_sweep", _stub_run_parameter_sweep)
    monkeypatch.setattr(
        "pa_core.reporting.sweep_excel.export_sweep_results",
        _stub_export_sweep_results,
    )
    monkeypatch.setattr(
        "pa_core.reporting.export_packet.create_export_packet",
        _stub_create_export_packet,
    )

    repo_root = Path(__file__).resolve().parents[1]
    idx_csv = repo_root / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "sweep.xlsx"

    main(
        [
            "--config",
            str(config_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--prev-manifest",
            str(prev_manifest_path),
            "--packet",
        ]
    )

    assert "prev_summary_df" in captured
    assert "prev_manifest" in captured
    pd.testing.assert_frame_equal(captured["prev_summary_df"], prev_summary)
    assert captured["prev_manifest"] == prev_manifest
