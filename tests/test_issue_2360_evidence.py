"""Acceptance must reject stale, skipped, or incomplete historical receipts."""

import json
import shutil
from pathlib import Path

import pytest

from scripts import verify_issue_2360_evidence as verifier


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    destination = tmp_path / "evidence"
    shutil.copytree(verifier.DEFAULT, destination)
    return destination


def change_json(directory: Path, name: str, update) -> None:
    path = directory / name
    record = json.loads(path.read_text())
    update(record)
    path.write_text(json.dumps(record))


def test_recovered_original_receipts_are_valid() -> None:
    assert verifier.check_revisions(verifier.DEFAULT)[0] == "PASS"
    assert verifier.check_ci(verifier.DEFAULT)[0] == "PASS"
    assert verifier.check_mutations(verifier.DEFAULT)[0] == "PASS"
    assert verifier.check_contract(verifier.DEFAULT)[0] == "PASS"


@pytest.mark.parametrize("breakage", ["stale", "skipped", "missing_log"])
def test_ci_requires_exact_head_and_executed_pytest(evidence: Path, breakage: str) -> None:
    if breakage == "stale":
        change_json(evidence, "recheck/ci-run.json", lambda run: run.update(head_sha="0" * 40))
    elif breakage == "skipped":
        change_json(
            evidence,
            "recheck/ci-jobs.json",
            lambda record: record.update(
                jobs=[{"run_id": 41661022529, "status": "completed", "conclusion": "skipped"}]
            ),
        )
    else:
        (evidence / "recheck/ci-pytest-excerpt.txt").write_text("no executed pytest receipt")
    assert verifier.check_ci(evidence)[0] == "FAIL"


@pytest.mark.parametrize("breakage", ["missing", "duplicate", "wrong_failure", "not_restored"])
def test_mutations_require_each_original_named_failure_and_restoration(
    evidence: Path, breakage: str
) -> None:
    def update(record):
        rows = record["results"]
        if breakage == "missing":
            rows.pop()
        elif breakage == "duplicate":
            rows[-1] = rows[0]
        elif breakage == "wrong_failure":
            (evidence / "recheck/mutations" / rows[0]["red_output"]).write_text(
                "FAILED tests/unrelated.py::test_other\n1 failed"
            )
        else:
            rows[0]["restored_sha256"] = "0" * 64

    change_json(evidence, "recheck/mutations/mutation-replay.json", update)
    assert verifier.check_mutations(evidence)[0] == "FAIL"


def test_missing_or_malformed_receipts_fail_without_crashing(evidence: Path) -> None:
    (evidence / "recheck/ci-run.json").write_text("{")
    (evidence / "recheck/mutations/mutation-replay.json").unlink()
    results = verifier.verify(evidence)
    assert results["CI"][0] == results["mutations"][0] == "FAIL"


@pytest.mark.parametrize("requirement", ["CI", "mutations"])
def test_missing_mandatory_proof_prevents_overall_pass(
    evidence: Path, requirement: str, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # Isolate each mandatory check: unrelated UNKNOWN findings cannot mask a bug.
    selected = verifier.CHECKS[requirement]
    monkeypatch.setattr(verifier, "CHECKS", {requirement: selected})
    assert verifier.main(["--evidence-dir", str(evidence)]) == 0
    assert "OVERALL PASS" in capsys.readouterr().out
    receipt = "ci-run.json" if requirement == "CI" else "mutations/mutation-replay.json"
    (evidence / "recheck" / receipt).unlink()
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "OVERALL FAIL" in output
    assert "OVERALL PASS" not in output


def test_ranking_unknown_is_not_reported_as_historical_compliance() -> None:
    status, detail = verifier.check_ranking(verifier.DEFAULT)
    assert status == "PASS"  # The contract permits an explicit UNKNOWN disclosure.
    assert "UNKNOWN" in detail
    assert "no selection compliance claim" in detail


@pytest.mark.parametrize(
    "breakage", [None, "later_revision", "selection", "denominator", "shortfall"]
)
def test_coverage_requires_isolated_comparable_execution(evidence: Path, breakage: str) -> None:
    # This synthetic pair exercises validation; it is never written to durable evidence.
    entries = []
    for index, revision in enumerate((verifier.BASE, verifier.MERGED)):
        name = f"synthetic-{index}.json"
        summary = {
            "num_statements": 100,
            "num_branches": 40,
            "missing_lines": 80 if index == 0 else 20,
            "missing_branches": 30 if index == 0 else 20,
        }
        if index == 1 and breakage == "denominator":
            summary["num_statements"] += 1
        if index == 1 and breakage == "shortfall":
            summary["missing_lines"] = 75
        (evidence / name).write_text(
            json.dumps({"files": {"dashboard/pages/3_Scenario_Wizard.py": {"summary": summary}}})
        )
        entries.append(
            {
                "revision": revision,
                "exit_code": 0,
                "selection": "tests/ -m 'not slow and not live_llm'",
                "settings": {"branch": True},
                "coverage_record": name,
            }
        )
    if breakage == "later_revision":
        entries[1]["revision"] = "cc7042594111cf6dc7ca4728c72e7ab271a0330f"
    if breakage == "selection":
        entries[1]["selection"] = "tests/test_wizard_render_coverage.py"
    (evidence / "recheck/coverage-comparison.json").write_text(
        json.dumps({"status": "measured", "before": entries[0], "after": entries[1]})
    )
    assert verifier.check_coverage(evidence)[0] == ("PASS" if breakage is None else "FAIL")
