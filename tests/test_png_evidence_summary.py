"""Coverage evidence must reconcile counts and reject incomplete captures."""

import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def evidence(tmp_path):
    path = (
        Path(__file__).resolve().parents[1] / "docs/evidence/issue-2336-png-failures/summarize.py"
    )
    spec = importlib.util.spec_from_file_location("png_evidence_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = dict(covered_lines=3, num_statements=4, missing_lines=1, percent_covered=75.0)
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "totals": summary,
                "files": {"pa_core/example.py": {"summary": summary, "missing_lines": [9]}},
            }
        )
    )
    junit = tmp_path / "junit.xml"
    suite = ET.Element("testsuite", tests="3", failures="1", errors="0", skipped="1")
    for name in ("passing", "failing", "skipped"):
        node = ET.SubElement(suite, "testcase", classname="tests.test_example", name=name)
        if name == "failing":
            ET.SubElement(node, "failure", message="AssertionError: mismatch")
        elif name == "skipped":
            ET.SubElement(node, "skipped")
    ET.ElementTree(suite).write(junit)
    return module, coverage, junit


def test_summary_preserves_exact_totals_and_named_failures(evidence):
    module, coverage, junit = evidence
    report = module.summarize(coverage, junit, 1)
    assert report["counts"] == dict(passed=1, failed=1, errors=0, skipped=1)
    assert report["totals"] == json.loads(coverage.read_text())["totals"]
    assert report["failures"] == [
        {
            "node": "tests/test_example.py::failing",
            "outcome": "failed",
            "message": "AssertionError: mismatch",
        }
    ]
    assert report["largest_gaps"][0]["path"] == "pa_core/example.py"
    assert report["modules"][0]["missing_line_numbers"] == [9]


@pytest.mark.parametrize("returncode", [2, 124, 130, -9])
def test_summary_rejects_interrupted_runs(evidence, returncode):
    module, coverage, junit = evidence
    with pytest.raises(ValueError, match="Incomplete pytest run"):
        module.summarize(coverage, junit, returncode)


@pytest.mark.parametrize("key", ["covered_lines", "num_statements", "missing_lines"])
def test_summary_rejects_inconsistent_coverage_totals(evidence, key):
    module, coverage, junit = evidence
    data = json.loads(coverage.read_text())
    data["totals"][key] += 1
    coverage.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f"totals.{key}"):
        module.summarize(coverage, junit, 1)


@pytest.mark.parametrize("attribute", ["tests", "failures", "errors", "skipped"])
def test_summary_rejects_partial_or_inconsistent_junit(evidence, attribute):
    module, coverage, junit = evidence
    tree = ET.parse(junit)
    root = tree.getroot()
    root.set(attribute, str(int(root.get(attribute)) + 1))
    tree.write(junit)
    with pytest.raises(ValueError, match="JUnit"):
        module.summarize(coverage, junit, 1)


def test_summary_rejects_success_exit_with_failed_tests(evidence):
    module, coverage, junit = evidence
    with pytest.raises(ValueError, match="exit disagrees"):
        module.summarize(coverage, junit, 0)
