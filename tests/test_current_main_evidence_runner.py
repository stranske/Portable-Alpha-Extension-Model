"""Focused checks for evidence isolation and subprocess/restoration failure paths."""

import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/issue-2336-png-failures/current-main-20261111"
)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timeout_kills_child_process_group(tmp_path, monkeypatch):
    runner = load_script("run_baseline")
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "TIMEOUT_SECONDS", 1)
    pid_file = tmp_path / "child.pid"
    heartbeat = tmp_path / "heartbeat"
    child = (
        "import time; from pathlib import Path\n"
        "while True:\n"
        f"    Path({str(heartbeat)!r}).write_text(str(time.monotonic_ns()))\n"
        "    time.sleep(0.02)\n"
    )
    parent = (
        "import subprocess, sys, time; from pathlib import Path; "
        f"p = subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    child_pid = None
    try:
        result = runner.run("timeout", [sys.executable, "-c", parent])
        child_pid = int(pid_file.read_text())
        assert result["timeout"] and result["exit"] is None
        time.sleep(0.1)
        last_heartbeat = heartbeat.read_text()
        time.sleep(0.2)
        assert heartbeat.read_text() == last_heartbeat, f"child {child_pid} still running"
    finally:
        if child_pid is None and pid_file.exists():
            child_pid = int(pid_file.read_text())
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_success_then_collection_failure_removes_old_coverage(tmp_path, monkeypatch):
    runner = load_script("run_baseline")
    root, out = tmp_path / "root", tmp_path / "out"
    root.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUT", out)
    monkeypatch.setattr(runner, "PRODUCTION", [])
    monkeypatch.setattr(runner, "git", lambda *args: "test-head")
    monkeypatch.setattr(runner, "versions", lambda: {})
    success = True

    def run(name, argv):
        assert not (root / "coverage.json").exists()
        assert not (out / "coverage.json").exists() or name == "focused"
        if name == "full" and success:
            (root / "coverage.json").write_text('{"fresh": true}')
        return {"name": name, "exit": 0 if success else 2}

    monkeypatch.setattr(runner, "run", run)
    assert runner.main() == 0
    assert json.loads((out / "coverage.json").read_text()) == {"fresh": True}
    (root / "coverage.json").write_text("stale root report")
    success = False
    assert runner.main() == 0
    assert not (out / "coverage.json").exists()
    assert not (root / "coverage.json").exists()
    record = json.loads((out / "run.json").read_text())
    assert record["stale_coverage_removed"]
    assert record["stale_output_coverage_removed"]
    assert not record["coverage_generated_by_this_run"]
    assert record["runs"][0]["exit"] == 2


def test_restoration_records_survive_mutation_pair_failure(tmp_path, monkeypatch):
    replay = load_script("replay_current")
    out = tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", ["replay_current.py", str(out)])
    monkeypatch.setattr(
        replay, "run_path", lambda name, *args: {"path": name, "restored_equals_baseline": True}
    )

    def fail_pairs():
        raise RuntimeError("pair execution failed")

    monkeypatch.setattr(replay, "load_replay", lambda: SimpleNamespace(main=fail_pairs))
    with pytest.raises(RuntimeError, match="pair execution failed"):
        replay.main()
    records = json.loads((out / "restoration-paths.json").read_text())
    assert [p["path"] for p in records] == ["timeout-path", "failure-path"]
