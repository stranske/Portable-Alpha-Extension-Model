import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

yaml: Any = pytest.importorskip("yaml")

sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
pptx_mod = types.ModuleType("pptx")
pptx_util = types.ModuleType("pptx.util")
pptx_mod.Presentation = object  # type: ignore[attr-defined]
pptx_util.Inches = lambda x: x  # type: ignore[attr-defined]
pptx_mod.util = pptx_util  # type: ignore[attr-defined]
sys.modules.setdefault("pptx", pptx_mod)
sys.modules.setdefault("pptx.util", pptx_util)

from pa_core.cli import main  # noqa: E402


@pytest.fixture(autouse=True)
def fast_parameter_sweep(monkeypatch):
    def _run_parameter_sweep(*_args, **_kwargs):
        return []

    def _export_sweep_results(*_args, **_kwargs):
        return None

    monkeypatch.setattr("pa_core.sweep.run_parameter_sweep", _run_parameter_sweep)
    monkeypatch.setattr("pa_core.reporting.sweep_excel.export_sweep_results", _export_sweep_results)


def test_manifest_written(tmp_path):
    cfg = {"N_SIMULATIONS": 1, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"
    seed = 123

    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--seed",
            str(seed),
        ]
    )

    manifest_path = out_file.with_name("manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == seed
    assert manifest["config"]["N_SIMULATIONS"] == 1
    assert str(cfg_path) in manifest["data_files"]
    assert manifest["backend"] == "numpy"


def test_manifest_records_run_log(tmp_path, monkeypatch):
    cfg = {"N_SIMULATIONS": 1, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"
    out_file = tmp_path / "out.xlsx"

    monkeypatch.chdir(tmp_path)
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file),
            "--log-json",
        ]
    )

    manifest = json.loads(out_file.with_name("manifest.json").read_text())
    run_log = Path(manifest["run_log"])
    if not run_log.is_absolute():
        run_log = tmp_path / run_log
    assert run_log.exists()


def test_manifest_records_previous_run(tmp_path):
    cfg = {"N_SIMULATIONS": 1, "N_MONTHS": 1}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    idx_csv = Path(__file__).resolve().parents[1] / "data" / "sp500tr_fred_divyield.csv"

    run1_dir = tmp_path / "run1"
    run1_dir.mkdir()
    out_file1 = run1_dir / "out.xlsx"
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file1),
        ]
    )
    manifest1 = out_file1.with_name("manifest.json")
    assert manifest1.exists()

    run2_dir = tmp_path / "run2"
    run2_dir.mkdir()
    out_file2 = run2_dir / "out.xlsx"
    main(
        [
            "--config",
            str(cfg_path),
            "--index",
            str(idx_csv),
            "--output",
            str(out_file2),
            "--prev-manifest",
            str(manifest1),
        ]
    )

    manifest2 = json.loads(out_file2.with_name("manifest.json").read_text())
    assert manifest2["previous_run"] == str(manifest1)
