import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pa_core.cli import Dependencies, main


def test_sweep_packet_passes_prev_diff(monkeypatch, tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "N_SIMULATIONS": 1,
                "N_MONTHS": 1,
                "financing_mode": "broadcast",
                "analysis_mode": "returns",
            }
        )
    )

    prev_summary = pd.DataFrame(
        {
            "Agent": ["Base"],
            "terminal_AnnReturn": [0.05],
            "monthly_AnnVol": [0.12],
            "terminal_ShortfallProb": [0.1],
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
                "terminal_AnnReturn": [0.06],
                "monthly_AnnVol": [0.11],
                "terminal_ShortfallProb": [0.08],
            }
        )
        return [{"summary": summary, "combination_id": 1}]

    def _stub_export_sweep_results(_results, filename="Sweep.xlsx", **_kwargs):
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


def test_packet_includes_stress_delta(monkeypatch, tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "N_SIMULATIONS": 1,
                "N_MONTHS": 2,
                "financing_mode": "broadcast",
                "analysis_mode": "single_with_sensitivity",
            }
        )
    )

    captured: dict[str, object] = {}

    def _stub_create_export_packet(**kwargs):
        captured.update(kwargs)
        return (str(tmp_path / "out.pptx"), str(tmp_path / "out.xlsx"))

    class _StubFig:
        def write_image(self, *_args, **_kwargs):
            return None

    fig = _StubFig()
    viz_stub = types.ModuleType("pa_core.viz")
    viz_stub.risk_return = types.SimpleNamespace(make=lambda _df: fig)
    viz_stub.sharpe_ladder = types.SimpleNamespace(make=lambda _df: fig)
    viz_stub.sunburst = types.SimpleNamespace(make=lambda _df: fig)
    monkeypatch.setitem(sys.modules, "pa_core.viz", viz_stub)
    monkeypatch.setattr(
        "pa_core.reporting.export_packet.create_export_packet",
        _stub_create_export_packet,
    )
    monkeypatch.setattr(
        "pa_core.sim.sensitivity.one_factor_deltas",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    deps = Dependencies(
        build_from_config=lambda _cfg: object(),
        export_to_excel=lambda *_args, **_kwargs: None,
        draw_financing_series=lambda *_args, **_kwargs: (
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        ),
        draw_joint_returns=lambda *_args, **_kwargs: (
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        ),
        build_cov_matrix=lambda *_args, **_kwargs: np.zeros((4, 4)),
        simulate_agents=lambda *_args, **_kwargs: {"Base": np.array([[0.01, 0.02]])},
    )

    repo_root = Path(__file__).resolve().parents[1]
    idx_csv = repo_root / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"

    main(
        [
            "--config",
            str(config_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--stress-preset",
            "liquidity_squeeze",
            "--packet",
            "--sensitivity",
        ],
        deps=deps,
    )

    assert "stress_delta_df" in captured
    stress_delta_df = captured["stress_delta_df"]
    assert isinstance(stress_delta_df, pd.DataFrame)
    assert not stress_delta_df.empty
    assert "Agent" in stress_delta_df.columns


def test_stress_delta_written_to_output_workbook(monkeypatch, tmp_path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "N_SIMULATIONS": 1,
                "N_MONTHS": 2,
                "financing_mode": "broadcast",
                "analysis_mode": "single_with_sensitivity",
            }
        )
    )

    def _stub_export_to_excel(
        _inputs_dict,
        summary,
        _raw_returns_dict,
        filename="Outputs.xlsx",
        **_kwargs,
    ):
        summary.to_excel(filename, sheet_name="Summary", index=False)

    deps = Dependencies(
        build_from_config=lambda _cfg: object(),
        export_to_excel=_stub_export_to_excel,
        draw_financing_series=lambda *_args, **_kwargs: (
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        ),
        draw_joint_returns=lambda *_args, **_kwargs: (
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
        ),
        build_cov_matrix=lambda *_args, **_kwargs: np.zeros((4, 4)),
        simulate_agents=lambda *_args, **_kwargs: {"Base": np.array([[0.01, 0.02]])},
    )

    repo_root = Path(__file__).resolve().parents[1]
    idx_csv = repo_root / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"

    monkeypatch.setattr(
        "pa_core.sim.sensitivity.one_factor_deltas",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "pa_core.sensitivity.one_factor_deltas",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    main(
        [
            "--config",
            str(config_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--stress-preset",
            "liquidity_squeeze",
            "--sensitivity",
        ],
        deps=deps,
    )

    stress_sheet = pd.read_excel(out_file, sheet_name="StressDelta")
    base_sheet = pd.read_excel(out_file, sheet_name="BaseSummary")
    stressed_sheet = pd.read_excel(out_file, sheet_name="StressedSummary")
    assert "Agent" in stress_sheet.columns
    assert "Agent" in base_sheet.columns
    assert "Agent" in stressed_sheet.columns
