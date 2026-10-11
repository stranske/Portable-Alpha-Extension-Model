"""Mutation replay must never write the caller's source, including on interruption."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
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


@pytest.mark.parametrize("interruption", ["timeout", "keyboard"])
def test_snapshot_replay_restores_source_after_phase_failure(
    checkout, tmp_path, monkeypatch, interruption
):
    active, files = checkout
    path = Path(__file__).resolve().parents[1] / "docs/evidence/issue-2336-cli-snapshots/replay.py"
    spec = importlib.util.spec_from_file_location("snapshot_cli_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, test = active / files[0], active / files[1]
    original, original_test = source.read_bytes(), test.read_bytes()
    # Interrupt the first RED, its restored GREEN, then the next case's RED.
    # Receipts must retain completed phases without claiming the interrupted one.
    for fail_at in range(3):
        output = tmp_path / f"snapshot-output-{fail_at}"
        output.mkdir()
        calls = []

        def fail(argv, *, cwd, **kwargs):
            assert cwd == active
            index = len(calls)
            calls.append(argv)
            red = index % 2 == 0
            assert (source.read_bytes() != original) == red
            if index == fail_at:
                if interruption == "timeout":
                    raise subprocess.TimeoutExpired(argv, 90)
                raise KeyboardInterrupt
            name = argv[3].split("::", 1)[1]
            suite = ET.Element("testsuite")
            result = ET.SubElement(suite, "testcase", name=name)
            if red:
                ET.SubElement(result, "failure", message="simulated production regression")
            junit = Path(
                next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--junitxml="))
            )
            ET.ElementTree(suite).write(junit, encoding="utf-8")
            return subprocess.CompletedProcess(argv, 1 if red else 0)

        monkeypatch.setattr(module.subprocess, "run", fail)
        expected = subprocess.TimeoutExpired if interruption == "timeout" else KeyboardInterrupt
        with pytest.raises(expected):
            module.replay(active, output, sys.executable)

        assert len(calls) == fail_at + 1
        assert source.read_bytes() == original
        assert test.read_bytes() == original_test
        controls = json.loads((output / "controls.json").read_text(encoding="utf-8"))
        assert controls["source_sha256"] == module.digest(original)
        assert controls["restored_sha256"] == module.digest(original)
        assert controls["test_sha256"] == module.digest(original_test)
        assert len(controls["cases"]) == fail_at // 2 + 1
        phases = [phase for case in controls["cases"] for phase in case["phases"]]
        assert len(phases) == fail_at
        for index, phase in enumerate(phases):
            red = index % 2 == 0
            assert phase["phase"] == ("red" if red else "green")
            assert phase["exit"] == (1 if red else 0)
            case = controls["cases"][index // 2]
            assert phase["source_sha256"] == (
                case["mutant_sha256"] if red else controls["source_sha256"]
            )
            assert phase["argv"] == calls[index]
            assert Path(phase["cwd"]) == active
            assert (output / phase["console"]).is_file()
            assert (output / phase["junit"]).is_file()
        assert controls["cases"][-1]["phases"] == (phases if fail_at == 1 else [])
        pending = f"{fail_at // 2:02d}-{'red' if fail_at % 2 == 0 else 'green'}"
        assert (output / f"{pending}.txt").is_file()
        assert not (output / f"{pending}.xml").exists()


@pytest.mark.parametrize("interruption", ["timeout", "keyboard"])
@pytest.mark.parametrize("cleanup_failure", ["cache_removed", "cache_denied", "source_read"])
def test_snapshot_cleanup_preserves_original_error_and_partial_receipt(
    checkout, tmp_path, monkeypatch, interruption, cleanup_failure
):
    active, files = checkout
    path = Path(__file__).resolve().parents[1] / "docs/evidence/issue-2336-cli-snapshots/replay.py"
    spec = importlib.util.spec_from_file_location("snapshot_cleanup_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = active / files[0]
    original = source.read_bytes()
    output = tmp_path / "cleanup-output"
    output.mkdir()
    interrupted = False
    cache = active / "pa_core/__pycache__/cli.concurrent.pyc"
    original_unlink, original_read = Path.unlink, Path.read_bytes
    expected = (
        subprocess.TimeoutExpired(["pytest"], 90)
        if interruption == "timeout"
        else KeyboardInterrupt()
    )

    def fail(argv, **kwargs):
        nonlocal interrupted
        cache.write_bytes(b"cache created during interrupted phase")
        interrupted = True
        raise expected

    def unlink(item, *args, **kwargs):
        if interrupted and item == cache:
            if cleanup_failure == "cache_denied":
                raise PermissionError("injected cache cleanup failure")
            if cleanup_failure == "cache_removed" and item.exists():
                original_unlink(item)
        return original_unlink(item, *args, **kwargs)

    def read(item):
        if interrupted and item == source and cleanup_failure == "source_read":
            raise OSError("injected restoration read failure")
        return original_read(item)

    with monkeypatch.context() as patch:
        patch.setattr(module.subprocess, "run", fail)
        patch.setattr(Path, "unlink", unlink)
        patch.setattr(Path, "read_bytes", read)
        with pytest.raises(type(expected)) as caught:
            module.replay(active, output, sys.executable)
        assert caught.value is expected

    assert source.read_bytes() == original
    controls = json.loads((output / "controls.json").read_text())
    assert controls["source_sha256"] == module.digest(original)
    assert controls["cases"][0]["phases"] == []
    if cleanup_failure == "source_read":
        assert controls["restored_sha256"] is None
        assert controls["cleanup_errors"][0]["stage"] == "read_restored_source"
    else:
        assert controls["restored_sha256"] == module.digest(original)
        if cleanup_failure == "cache_denied":
            assert controls["cleanup_errors"][0]["stage"] == "remove_cache"
        else:
            assert controls.get("cleanup_errors", []) == []
