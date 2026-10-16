"""Keep the PPTX evidence acceptance and output-reuse controls enforced in CI."""

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPLAY = Path(__file__).resolve().parents[1] / "docs/evidence/issue-2336-pptx-failures/replay.py"
EXPECTED_NAME = "test_single_run_pptx_failure_preserves_results[renderer]"
CHAINED = (
    "tests/test_cli_pptx_failures.py:151: AssertionError\n"
    "During handling of the above exception, another exception occurred:\n"
    "pa_core/cli.py:99: ValueError: boom"
)
ASSERTION = '<failure message="AssertionError: missing ERROR diagnostic" />'


@pytest.fixture
def replay():
    spec = importlib.util.spec_from_file_location("pptx_failure_replay", REPLAY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(body="", *, name=EXPECTED_NAME, module="tests.test_cli_pptx_failures"):
    return (
        '<testsuites><testsuite><testcase classname="'
        + module
        + '" name="'
        + name
        + '">'
        + body
        + "</testcase></testsuite></testsuites>"
    )


@pytest.mark.parametrize(
    "phase,returncode,body",
    [
        ("red", 1, ASSERTION),
        ("red", 1, "<failure>tests/test_cli_pptx_failures.py:151: AssertionError</failure>"),
        ("green", 0, ""),
    ],
    ids=["assertion-message", "pytest-assertion-traceback", "restored-pass"],
)
def test_pptx_proof_accepts_exact_node_outcomes(replay, tmp_path, phase, returncode, body):
    path = tmp_path / "result.xml"
    path.write_text(report(body))
    replay.validate_junit_proof(path, EXPECTED_NAME, phase, returncode, "renderer")


@pytest.mark.parametrize(
    "xml,phase,returncode",
    [
        (report(ASSERTION, name="another-test"), "red", 1),
        (report(ASSERTION, module="tests.test_cli_png_failures"), "red", 1),
        ("<testsuites><testsuite /></testsuites>", "red", 1),
        (report(ASSERTION).replace("</testsuite>", "<testcase /></testsuite>"), "red", 1),
        (report(ASSERTION + '<error message="setup failed" />'), "red", 1),
        (report(ASSERTION).replace("</testsuite>", "<error /></testsuite>"), "red", 1),
        (report(ASSERTION + "<skipped />"), "red", 1),
        (report("<skipped />"), "green", 0),
        (report(), "red", 1),
        (report('<failure message="ValueError: AssertionError: decoy" />'), "red", 1),
        (report(ASSERTION + ASSERTION), "red", 1),
        (report("<failure>" + CHAINED + "</failure>"), "red", 1),
        (report(ASSERTION), "red", 0),
        (report(ASSERTION), "red", 2),
        (report(ASSERTION), "green", 0),
        (report(), "green", 1),
    ],
    ids=[
        "wrong-node",
        "wrong-module",
        "empty",
        "extra-node",
        "setup-error",
        "collection-error",
        "skipped-red",
        "skipped-green",
        "red-without-failure",
        "non-assertion-decoy",
        "duplicate-failures",
        "chained-final-non-assertion",
        "red-exit-zero",
        "red-exit-two",
        "green-failure",
        "green-nonzero-exit",
    ],
)
def test_pptx_proof_rejects_invalid_outcomes(replay, tmp_path, xml, phase, returncode):
    path = tmp_path / "result.xml"
    path.write_text(xml)
    with pytest.raises(RuntimeError):
        replay.validate_junit_proof(path, EXPECTED_NAME, phase, returncode, "renderer")


def test_pptx_proof_rejects_truncated_junit(replay, tmp_path):
    path = tmp_path / "result.xml"
    path.write_text("<testsuites><testsuite><testcase")
    with pytest.raises(ET.ParseError):
        replay.validate_junit_proof(path, EXPECTED_NAME, "red", 1, "renderer")


def test_pptx_proof_refuses_output_reuse_before_archiving(replay, tmp_path, monkeypatch):
    output = tmp_path / "existing-proof"
    output.mkdir()
    sentinel = output / "renderer-red.xml"
    sentinel.write_bytes(b"existing proof must survive")

    def unexpected_archive(*args, **kwargs):
        pytest.fail("Output reuse must be refused before Git archiving")

    monkeypatch.setattr(sys, "argv", [str(REPLAY), "--output", str(output)])
    monkeypatch.setattr(replay.subprocess, "check_output", unexpected_archive)
    with pytest.raises(FileExistsError):
        replay.main()

    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"existing proof must survive"


def test_pptx_proof_refuses_mismatched_caller_before_archive_or_mutation(
    replay, tmp_path, monkeypatch
):
    output = tmp_path / "new-proof"

    def unexpected(*args, **kwargs):
        pytest.fail("Caller mismatch must be refused before archiving or mutation")

    monkeypatch.setattr(sys, "argv", [str(REPLAY), "--output", str(output)])
    monkeypatch.setattr(replay, "SOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(replay.subprocess, "check_output", unexpected)
    monkeypatch.setattr(replay.subprocess, "run", unexpected)
    with pytest.raises(RuntimeError, match="Caller"):
        replay.main()

    assert not output.exists()
