"""JUnit outcome checks for the PNG failure replay driver."""

import importlib.util
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
