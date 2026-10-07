"""Mutation replay must never write the caller's source, including on interruption."""

import importlib.util
import json
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
    cache = active / "pa_core/__pycache__/cli.cpython-312.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"existing caller cache")
    return active, files


def checkout_bytes(active):
    return {
        path.relative_to(active): path.read_bytes() for path in active.rglob("*") if path.is_file()
    }


def assert_checkout_unchanged(active, original):
    current = checkout_bytes(active)
    assert current.keys() == original.keys()
    for relative, content in original.items():
        assert current[relative] == content, str(relative)


def test_existing_output_refused_before_copying_or_mutating_checkout(
    checkout, tmp_path, monkeypatch
):
    active, _ = checkout
    _, module = load_driver()
    original = checkout_bytes(active)
    output = tmp_path / "output"
    output.mkdir()
    proof = output / "controls.json"
    proof.write_bytes(b"existing proof must survive")
    nested = output / "phases"
    nested.mkdir()
    (nested / "console.txt").write_bytes(b"prior phase output must survive")
    output_bytes = checkout_bytes(output)
    monkeypatch.setattr(module, "__file__", str(active / REPLAY))
    monkeypatch.setattr(sys, "argv", ["replay.py", "--output", str(output)])

    def cannot_start(*args, **kwargs):
        pytest.fail("Existing proof must be refused before snapshotting or running mutations")

    monkeypatch.setattr(module.subprocess, "check_output", cannot_start)
    monkeypatch.setattr(module.subprocess, "run", cannot_start)
    with pytest.raises(FileExistsError):
        module.main()
    assert_checkout_unchanged(output, output_bytes)
    assert nested.is_dir()
    assert_checkout_unchanged(active, original)


@pytest.mark.parametrize("interruption", ["timeout", "keyboard"])
def test_phase_failure_mutates_only_private_tree_and_preserves_active_source(
    checkout, tmp_path, monkeypatch, interruption
):
    active, files = checkout
    _, module = load_driver()
    source = active / files[0]
    original = source.read_bytes()
    active_bytes = checkout_bytes(active)
    output = tmp_path / "output"
    monkeypatch.setattr(module, "__file__", str(active / REPLAY))
    monkeypatch.setattr(sys, "argv", ["replay.py", "--output", str(output)])
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *a, **kw: b"\0".join(str(p).encode() for p in files),
    )
    private = []
    mutant_hashes = []

    def fail(argv, *, cwd, **kwargs):
        assert cwd != active
        assert (cwd / files[0]).read_bytes() != original
        assert source.read_bytes() == original
        private.append(cwd)
        mutant_hashes.append(module.digest((cwd / files[0]).read_bytes()))
        if interruption == "timeout":
            raise subprocess.TimeoutExpired(argv, 90)
        raise KeyboardInterrupt

    monkeypatch.setattr(module.subprocess, "run", fail)
    expected = subprocess.TimeoutExpired if interruption == "timeout" else KeyboardInterrupt
    with pytest.raises(expected):
        module.main()
    assert private and all(not p.exists() for p in private)
    assert_checkout_unchanged(active, active_bytes)

    # Interrupted phases must keep restoration evidence without claiming completion.
    controls = json.loads((output / "controls.json").read_text(encoding="utf-8"))
    assert controls["source_sha256"] == module.digest(original)
    assert controls["restored_sha256"] == controls["source_sha256"]
    assert controls["test_sha256"] == module.digest(active_bytes[files[1]])
    assert len(controls["cases"]) == 1
    assert controls["cases"][0]["mutant_sha256"] == mutant_hashes[0]
    assert controls["cases"][0]["phases"] == []
    assert (output / "00-red.txt").is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGTERM contract")
def test_sigterm_after_mutation_leaves_active_source_unchanged(checkout, tmp_path):
    active, files = checkout
    root, _ = load_driver()
    original = (active / files[0]).read_bytes()
    active_bytes = checkout_bytes(active)
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
    assert_checkout_unchanged(active, active_bytes)
