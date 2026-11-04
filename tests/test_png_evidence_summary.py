"""Coverage evidence must reconcile counts and reject incomplete captures."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
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


@pytest.mark.parametrize("defect", [None, "missing", "size", "hash"])
def test_manifest_requires_retained_artifact_bytes(evidence, defect):
    module, coverage, _ = evidence
    artifact = coverage.parent / "completed-suite.txt"
    artifact.write_bytes(b"pytest output\n")
    manifest = coverage.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    artifact.name: {
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                }
            }
        )
    )
    if defect == "missing":
        artifact.unlink()
    elif defect == "size":
        artifact.write_bytes(b"truncated")
    elif defect == "hash":
        artifact.write_bytes(b"altered text!\n")
    if defect is None:
        module.validate_manifest(manifest)
    else:
        with pytest.raises(ValueError, match="artifact.*completed-suite.txt"):
            module.validate_manifest(manifest)


@pytest.mark.parametrize("missing_artifact", [False, True])
def test_summary_cli_verifies_manifest_before_writing_report(evidence, missing_artifact):
    module, coverage, junit = evidence
    content = coverage.read_bytes()
    manifest = coverage.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    coverage.name: {
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                }
            }
        )
    )
    if missing_artifact:
        coverage.unlink()
    output = coverage.parent / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            module.__file__,
            "--coverage",
            str(coverage),
            "--junit",
            str(junit),
            "--exit-code",
            "1",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if missing_artifact:
        assert result.returncode != 0
        assert "Missing evidence artifact: coverage.json" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "3 / 4 statements (75.0%)" in result.stdout
        assert json.loads(output.read_text())["counts"]["failed"] == 1
