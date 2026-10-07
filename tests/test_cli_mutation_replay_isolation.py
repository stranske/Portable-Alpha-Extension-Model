"""Mutation replay must never write the caller's source, including on interruption."""

import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPLAY = Path("docs/evidence/issue-2336-next-boundaries/replay.py")


def load_driver():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cli_replay", root / REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module


@pytest.fixture
def checkout(tmp_path):
    root, _ = load_driver()
    active = tmp_path / "active"
    files = [Path("pa_core/cli.py"), Path("tests/test_cli_observability_boundaries.py")]
    for relative in files:
        target = active / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    return active, files


def test_phase_failure_mutates_only_private_tree_and_preserves_active_source(
    checkout, tmp_path, monkeypatch
):
    active, files = checkout
    _, module = load_driver()
    source = active / files[0]
    original = source.read_bytes()
    monkeypatch.setattr(module, "__file__", str(active / REPLAY))
    monkeypatch.setattr(sys, "argv", ["replay.py", "--output", str(tmp_path / "output")])
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *a, **kw: b"\0".join(str(p).encode() for p in files),
    )
    private = []

    def fail(argv, *, cwd, **kwargs):
        assert cwd != active
        assert (cwd / files[0]).read_bytes() != original
        assert source.read_bytes() == original
        private.append(cwd)
        raise subprocess.TimeoutExpired(argv, 90)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        module.main()
    assert private and all(not p.exists() for p in private)
    assert source.read_bytes() == original


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGTERM contract")
def test_sigterm_after_mutation_leaves_active_source_unchanged(checkout, tmp_path):
    active, files = checkout
    root, _ = load_driver()
    original = (active / files[0]).read_bytes()
    marker = tmp_path / "private-tree.txt"
    helper = tmp_path / "interrupt.py"
    helper.write_text(
        "import importlib.util, os, signal\n"
        "from pathlib import Path\n"
        f"spec = importlib.util.spec_from_file_location('driver', {str(root / REPLAY)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"module.__file__ = {str(active / REPLAY)!r}\n"
        f"module.subprocess.check_output = lambda *a, **kw: {b'pa_core/cli.py' + bytes([0]) + b'tests/test_cli_observability_boundaries.py'!r}\n"
        "def interrupt(argv, *, cwd, **kwargs):\n"
        f"    Path({str(marker)!r}).write_text(str(cwd))\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "module.subprocess.run = interrupt\n"
        "module.main()\n"
    )
    proc = subprocess.run(
        [sys.executable, str(helper), "--output", str(tmp_path / "output")],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert proc.returncode == -signal.SIGTERM, proc.stderr
    private = Path(marker.read_text())
    assert private != active
    assert (private / files[0]).read_bytes() != original
    assert (active / files[0]).read_bytes() == original
