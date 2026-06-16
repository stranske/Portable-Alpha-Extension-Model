from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pa_core.contracts import (
    MANIFEST_CONTRACT,
    MANIFEST_FORMAT,
    MANIFEST_REQUIRED_FIELDS,
    RUN_END_FILENAME,
    RUN_END_MANIFEST_PATH_KEY,
    RUN_LOG_FILENAME,
    RUNS_DIR_NAME,
    SUMMARY_CVAR_CI95_HIGH_COLUMN,
    SUMMARY_CVAR_CI95_LOW_COLUMN,
    SUMMARY_CVAR_COLUMN,
    SUMMARY_CVAR_SE_COLUMN,
    SUMMARY_CVAR_TERMINAL_COLUMN,
    SUMMARY_COLUMN_TYPES,
    SUMMARY_REQUIRED_COLUMNS,
    SUMMARY_SHEET_NAME,
    SUMMARY_TERMINAL_CVAR_CI95_HIGH_COLUMN,
    SUMMARY_TERMINAL_CVAR_CI95_LOW_COLUMN,
    SUMMARY_TERMINAL_CVAR_HALF_SAMPLE_DELTA_COLUMN,
    SUMMARY_TERMINAL_CVAR_SE_COLUMN,
    manifest_path_for_output,
    manifest_path_from_run_end,
    validate_manifest_payload,
    validate_run_directory,
    validate_summary_frame,
)
from pa_core.manifest import ManifestWriter
from pa_core.sim.metrics import summary_table


def test_validate_run_directory_happy_path(tmp_path: Path) -> None:
    run_dir = tmp_path / RUNS_DIR_NAME / "20200101T000000Z"
    run_dir.mkdir(parents=True)
    (run_dir / RUN_LOG_FILENAME).write_text('{"event":"run_start"}\n', encoding="utf-8")
    (run_dir / RUN_END_FILENAME).write_text('{"event":"run_end"}\n', encoding="utf-8")
    assert validate_run_directory(run_dir)


def test_validate_run_directory_missing_log(tmp_path: Path) -> None:
    run_dir = tmp_path / RUNS_DIR_NAME / "20200101T000000Z"
    run_dir.mkdir(parents=True)
    assert not validate_run_directory(run_dir)


def test_manifest_contract_fields(tmp_path: Path) -> None:
    assert MANIFEST_CONTRACT.filename.endswith(f".{MANIFEST_FORMAT}")
    assert MANIFEST_CONTRACT.run_end_manifest_key == RUN_END_MANIFEST_PATH_KEY

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("N_SIMULATIONS: 1\n", encoding="utf-8")
    data_path = tmp_path / "data.csv"
    data_path.write_text("a,b\n1,2\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    writer = ManifestWriter(manifest_path)
    writer.write(
        config_path=cfg_path,
        data_files=[data_path],
        seed=123,
        cli_args={"output": "Outputs.xlsx"},
        backend="numpy",
    )

    payload = json.loads(manifest_path.read_text())
    assert validate_manifest_payload(payload)
    for key in MANIFEST_REQUIRED_FIELDS:
        assert key in payload


def test_manifest_path_for_output() -> None:
    assert manifest_path_for_output("Outputs.xlsx").name == "manifest.json"


def test_manifest_path_from_run_end(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    run_end_path = tmp_path / RUN_END_FILENAME
    run_end_path.write_text(
        json.dumps({RUN_END_MANIFEST_PATH_KEY: str(manifest_path)}), encoding="utf-8"
    )
    assert manifest_path_from_run_end(run_end_path) == manifest_path


def test_summary_contract_matches_summary_table() -> None:
    returns = {
        "Base": np.array([[0.01, 0.02, 0.03]]),
        "A": np.array([[0.02, 0.01, 0.04]]),
    }
    summary = summary_table(returns, benchmark="Base")
    assert validate_summary_frame(summary)
    for col in SUMMARY_REQUIRED_COLUMNS:
        assert col in summary.columns
    assert set(SUMMARY_COLUMN_TYPES.keys()) == set(SUMMARY_REQUIRED_COLUMNS)
    assert SUMMARY_SHEET_NAME == "Summary"


def test_summary_table_populates_cvar_precision_contract_columns() -> None:
    base_returns = np.linspace(-0.10, 0.08, 1200, dtype=float).reshape(200, 6)
    active_returns = np.linspace(-0.12, 0.10, 1200, dtype=float).reshape(200, 6)
    returns = {
        "Base": base_returns,
        "A": active_returns,
    }

    summary = summary_table(returns, benchmark="Base")
    diagnostic_columns = [
        SUMMARY_CVAR_COLUMN,
        SUMMARY_CVAR_SE_COLUMN,
        SUMMARY_CVAR_CI95_LOW_COLUMN,
        SUMMARY_CVAR_CI95_HIGH_COLUMN,
        SUMMARY_CVAR_TERMINAL_COLUMN,
        SUMMARY_TERMINAL_CVAR_SE_COLUMN,
        SUMMARY_TERMINAL_CVAR_CI95_LOW_COLUMN,
        SUMMARY_TERMINAL_CVAR_CI95_HIGH_COLUMN,
        SUMMARY_TERMINAL_CVAR_HALF_SAMPLE_DELTA_COLUMN,
    ]

    assert diagnostic_columns == [col for col in SUMMARY_REQUIRED_COLUMNS if "CVaR" in col]
    assert summary[diagnostic_columns].notna().all().all()
    assert np.isfinite(summary[diagnostic_columns].to_numpy(dtype=float)).all()
