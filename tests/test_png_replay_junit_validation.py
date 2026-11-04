"""JUnit outcome checks for the PNG failure replay driver."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPLAY = Path(__file__).resolve().parents[1] / "docs/evidence/issue-2336-png-failures/replay.py"
EXPECTED_NAME = "test_single_run_png_failure_preserves_results[renderer]"


def load_replay():
    spec = importlib.util.spec_from_file_location("png_failure_replay", REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_junit(path, *, names, failures=(), errors=(), skipped=()):
    suite = ET.Element("testsuite")
    for index, name in enumerate(names):
        node = ET.SubElement(suite, "testcase", name=name)
        if index in failures:
            ET.SubElement(node, "failure", message="AssertionError: expected failure")
        if index in errors:
            ET.SubElement(node, "error", message="setup failed")
        if index in skipped:
            ET.SubElement(node, "skipped")
    ET.ElementTree(suite).write(path, encoding="utf-8")


def test_validate_junit_proof_accepts_red_and_green(tmp_path):
    module = load_replay()
    red = tmp_path / "red.xml"
    green = tmp_path / "green.xml"
    write_junit(red, names=[EXPECTED_NAME], failures=(0,))
    write_junit(green, names=[EXPECTED_NAME])
    module.validate_junit_proof(red, EXPECTED_NAME, "red", 1, "renderer")
    module.validate_junit_proof(green, EXPECTED_NAME, "green", 0, "renderer")


@pytest.mark.parametrize(
    "defect,returncode,phase",
    [
        ("extra-node", 1, "red"),
        ("wrong-name", 1, "red"),
        ("missing-node", 1, "red"),
        ("skipped", 1, "red"),
        ("error", 1, "red"),
        ("red-without-assertion", 1, "red"),
        ("red-exit-zero", 0, "red"),
        ("green-with-failure", 0, "green"),
        ("green-nonzero-exit", 1, "green"),
    ],
)
def test_validate_junit_proof_rejects_invalid_outcomes(tmp_path, defect, returncode, phase):
    module = load_replay()
    path = tmp_path / f"{defect}.xml"
    if defect == "extra-node":
        write_junit(path, names=[EXPECTED_NAME, "other"])
    elif defect == "wrong-name":
        write_junit(path, names=["other-name"], failures=(0,) if phase == "red" else ())
    elif defect == "missing-node":
        write_junit(path, names=[])
    elif defect == "skipped":
        write_junit(path, names=[EXPECTED_NAME], failures=(0,), skipped=(0,))
    elif defect == "error":
        write_junit(path, names=[EXPECTED_NAME], failures=(0,), errors=(0,))
    elif defect == "red-without-assertion":
        suite = ET.Element("testsuite")
        node = ET.SubElement(suite, "testcase", name=EXPECTED_NAME)
        ET.SubElement(node, "failure", message="ValueError: not an assertion")
        ET.ElementTree(suite).write(path, encoding="utf-8")
    elif defect == "red-exit-zero":
        write_junit(path, names=[EXPECTED_NAME], failures=(0,))
    elif defect == "green-with-failure":
        write_junit(path, names=[EXPECTED_NAME], failures=(0,))
    elif defect == "green-nonzero-exit":
        write_junit(path, names=[EXPECTED_NAME])
    else:
        raise AssertionError(defect)
    with pytest.raises(RuntimeError):
        module.validate_junit_proof(path, EXPECTED_NAME, phase, returncode, "renderer")


@pytest.mark.parametrize(
    "exception,accepted",
    [("assert False", True), ('raise ValueError("AssertionError: decoy")', False)],
)
def test_validate_junit_proof_reads_actual_pytest_assertion_type(tmp_path, exception, accepted):
    test = tmp_path / "test_actual.py"
    test.write_text("def test_actual():\n    " + exception + "\n")
    report = tmp_path / "actual.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            str(test),
            "--junitxml=" + str(report),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    module = load_replay()
    if accepted:
        module.validate_junit_proof(report, "test_actual", "red", result.returncode, "actual")
    else:
        with pytest.raises(RuntimeError):
            module.validate_junit_proof(report, "test_actual", "red", result.returncode, "actual")


@pytest.fixture
def revision_snapshot(tmp_path, monkeypatch):
    module = load_replay()
    repo = tmp_path / "caller"
    source = repo / "pa_core/cli.py"
    source.parent.mkdir(parents=True)
    original = (REPLAY.parents[3] / "pa_core/cli.py").read_bytes()
    source.write_bytes(original)
    test = repo / module.TEST
    test.parent.mkdir(parents=True)
    test.write_text("# caller's strengthened boundary test\n")
    monkeypatch.setattr(module, "__file__", str(repo / REPLAY.relative_to(REPLAY.parents[3])))
    revision = "a" * 40
    calls = []

    def git_output(argv, *, cwd):
        assert cwd == repo
        calls.append(argv)
        if argv[1] == "rev-parse":
            return (revision + "\n").encode()
        assert argv[1] == "archive"
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            for name, content in (("pa_core/cli.py", original), (module.TEST, b"# main test\n")):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                bundle.addfile(info, io.BytesIO(content))
        return archive.getvalue()

    monkeypatch.setattr(module.subprocess, "check_output", git_output)
    return module, source, original, revision, calls


def test_replay_rejects_caller_drift_before_running_mutations(
    revision_snapshot, tmp_path, monkeypatch
):
    module, source, original, _, _ = revision_snapshot
    source.write_bytes(original + b"\n# uncommitted production edit\n")
    output = tmp_path / "proof"
    monkeypatch.setattr(sys, "argv", [str(REPLAY), "--output", str(output)])

    def unexpected_run(*args, **kwargs):
        pytest.fail("A different caller revision must not be presented as current-source proof")

    monkeypatch.setattr(module.subprocess, "run", unexpected_run)
    with pytest.raises(RuntimeError, match="Caller production source differs"):
        module.main()
    assert source.read_bytes() == original + b"\n# uncommitted production edit\n"
    assert not (output / "results.json").exists()


def test_replay_freezes_selected_revision_and_records_restoration(
    revision_snapshot, tmp_path, monkeypatch
):
    module, source, original, revision, calls = revision_snapshot
    output = tmp_path / "proof"
    monkeypatch.setattr(
        sys, "argv", [str(REPLAY), "--output", str(output), "--revision", "origin/main"]
    )
    monkeypatch.setattr(module, "CASES", {"renderer": module.CASES["renderer"]})
    phases = []

    def run_phase(argv, *, cwd, **kwargs):
        private_source = (cwd / "pa_core/cli.py").read_bytes()
        assert source.read_bytes() == original
        assert cwd != source.parents[1]
        assert (cwd / module.TEST).read_bytes() == (source.parents[1] / module.TEST).read_bytes()
        report = Path(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--junitxml=")))
        red = not phases
        assert (private_source != original) == red
        write_junit(report, names=[EXPECTED_NAME], failures=(0,) if red else ())
        phases.append(private_source)
        return subprocess.CompletedProcess(argv, 1 if red else 0)

    monkeypatch.setattr(module.subprocess, "run", run_phase)
    module.main()
    assert calls == [
        ["git", "rev-parse", "--verify", "--end-of-options", "origin/main^{commit}"],
        ["git", "archive", revision],
    ]
    proof = json.loads((output / "results.json").read_text())
    assert proof["revision"] == revision
    assert proof["requested_revision"] == "origin/main"
    row = proof["results"][0]
    assert row["original_sha256"] == row["restored_sha256"] == module.digest(original)
    assert row["green"]["source_sha256"] == module.digest(original)
    assert row["red"]["source_sha256"] == module.digest(phases[0])
    assert source.read_bytes() == original
