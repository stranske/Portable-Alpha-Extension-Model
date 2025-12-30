from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from pa_core.logging_utils import JSONLogFormatter
from pa_core.manifest import ManifestWriter


def test_json_formatter_outputs_valid_json_line():
    logger = logging.getLogger("pa_core.test")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("hello", extra={"run_id": "abc123", "event": "unit"})

    line = stream.getvalue().strip()
    assert line, "no log output produced"
    obj = json.loads(line)
    assert obj["msg"] == "hello"
    assert obj["run_id"] == "abc123"
    assert obj["event"] == "unit"
    assert obj["level"] == "INFO"


def test_manifest_includes_backend_and_optional_run_log(tmp_path: Path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text("a: 1\n")
    data = tmp_path / "data.csv"
    data.write_text("x\n1\n")
    out = tmp_path / "manifest.json"

    mw = ManifestWriter(out)
    prev_manifest = tmp_path / "runs/20191231T235959Z/manifest.json"
    mw.write(
        config_path=cfg,
        data_files=[data],
        seed=42,
        cli_args={"output": "foo.xlsx"},
        backend="numpy",
        run_log=str(tmp_path / "runs/20200101T000000Z/run.log"),
        previous_run=prev_manifest,
    )
    obj = json.loads(out.read_text())
    assert obj.get("backend") == "numpy"
    assert obj.get("run_log", "").endswith("run.log")
    assert obj.get("previous_run") == str(prev_manifest)
