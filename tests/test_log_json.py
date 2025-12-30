import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pa_core.cli import main

yaml: Any = pytest.importorskip("yaml")


def test_log_json_creates_file_and_manifest(tmp_path, monkeypatch):
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
            "--seed",
            "123",
        ]
    )

    manifest_path = out_file.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    log_path = Path(manifest["run_log"])
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    first_line = lines[0]
    parsed = json.loads(first_line)
    assert {"level", "timestamp", "module", "message"} <= set(parsed)

    run_end = None
    for line in lines:
        obj = json.loads(line)
        if obj.get("event") == "run_end":
            run_end = obj
            break
    assert run_end is not None
    assert run_end.get("duration_seconds") is not None
    assert run_end.get("seed") == 123
    assert run_end.get("backend") == "numpy"
    assert isinstance(run_end.get("artifact_paths"), list)

    assert manifest.get("run_timing", {}).get("duration_seconds") is not None

    if shutil.which("jq"):
        msg = subprocess.check_output(
            ["jq", "-r", ".message"], input=first_line.encode()
        )
        assert msg.strip().decode() == parsed["message"]
