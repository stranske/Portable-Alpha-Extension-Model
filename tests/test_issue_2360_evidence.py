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
    assert verifier.check_coverage(verifier.DEFAULT)[0] == "PASS"
    assert verifier.check_contract(verifier.DEFAULT)[0] == "PASS"


@pytest.mark.parametrize("number", [2372, 2467])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_sha", "0" * 40),
        ("head_sha", "0" * 40),
        ("merge_sha", "0" * 40),
        ("files", ["tests/unrelated.py"]),
        ("url", "https://github.com/unrelated/repository/pull/1"),
        ("state", "open"),
        ("merged", False),
    ],
)
def test_revision_receipts_reject_changed_pr_metadata(
    evidence: Path, number: int, field: str, value: object, capsys
) -> None:
    def update(record):
        pr = next(pr for pr in record["pull_requests"] if pr["number"] == number)
        pr[field] = value

    change_json(evidence, "recheck/revisions.json", update)
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "FAIL: revisions:" in output
    assert "OVERALL FAIL" in output
    assert "OVERALL PASS" not in output


@pytest.mark.parametrize("breakage", ["issue_number", "issue_url", "duplicate_pr", "missing_pr"])
def test_revision_receipts_require_original_issue_and_unique_prs(
    evidence: Path, breakage: str
) -> None:
    def update(record):
        if breakage == "issue_number":
            record["issue"]["number"] = 2336
        elif breakage == "issue_url":
            record["issue"]["url"] = record["issue"]["url"].replace("stranske/", "unrelated/")
        elif breakage == "duplicate_pr":
            record["pull_requests"].append(record["pull_requests"][0].copy())
        else:
            record["pull_requests"].pop()

    change_json(evidence, "recheck/revisions.json", update)
    assert verifier.verify(evidence)["revisions"][0] == "FAIL"


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


@pytest.mark.parametrize(
    "breakage",
    [
        "stale_job_head",
        "other_attempt",
        "other_run",
        "unlisted_job",
        "failed_job",
        "skipped_pytest",
        "other_job_repository",
        "other_run_repository",
        "other_source",
    ],
)
def test_ci_rejects_mixed_run_and_job_receipts(evidence: Path, breakage: str) -> None:
    # A successful run and normalized job list must not mask a mismatched raw job.
    def update(record):
        job = record["job"]
        if breakage == "stale_job_head":
            job["head_sha"] = "0" * 40
        elif breakage == "other_attempt":
            job["run_attempt"] += 1
        elif breakage == "other_run":
            job["run_id"] += 1
        elif breakage == "unlisted_job":
            job["id"] += 1
            job["html_url"] = job["html_url"].rsplit("/", 1)[0] + f"/{job['id']}"
        elif breakage == "failed_job":
            job["conclusion"] = "failure"
        elif breakage == "skipped_pytest":
            for step in job["steps"]:
                if step["name"] == "Pytest (unit tests with coverage)":
                    step["conclusion"] = "skipped"
        elif breakage == "other_job_repository":
            job["html_url"] = job["html_url"].replace("stranske/", "unrelated/")
        elif breakage == "other_run_repository":
            change_json(
                evidence,
                "recheck/ci-run.json",
                lambda run: run.update(html_url=run["html_url"].replace("stranske/", "unrelated/")),
            )
        else:
            record["source_url"] = record["source_url"].replace("stranske/", "unrelated/")

    change_json(evidence, "recheck/ci-pytest-job.json", update)
    assert verifier.check_ci(evidence)[0] == "FAIL"


def test_missing_raw_ci_job_prevents_overall_pass(evidence: Path, capsys) -> None:
    (evidence / "recheck/ci-pytest-job.json").unlink()
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "FAIL: CI:" in output
    assert "OVERALL FAIL" in output
    assert "OVERALL PASS" not in output


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


@pytest.mark.parametrize("breakage", [None, "later_revision", "selection", "denominator"])
def test_coverage_requires_isolated_comparable_execution(evidence: Path, breakage: str) -> None:
    if breakage == "later_revision":
        change_json(
            evidence,
            "recheck/coverage-comparison.json",
            lambda record: record["after"].update(revision="cc704259"),
        )
    elif breakage == "selection":
        change_json(
            evidence,
            "recheck/coverage-comparison.json",
            lambda record: record["after"].update(selection="tests/test_wizard_render_coverage.py"),
        )
    elif breakage == "denominator":
        change_json(
            evidence,
            "recheck/coverage-merged.json",
            lambda data: data["files"]["dashboard/pages/3_Scenario_Wizard.py"]["summary"].update(
                num_statements=1051
            ),
        )
    assert verifier.verify(evidence)["coverage attribution"][0] == (
        "PASS" if breakage is None else "FAIL"
    )


@pytest.mark.parametrize(
    "breakage",
    [
        "line_only",
        "summary_only",
        "missing_count",
        "duplicate_line",
        "overlap_line",
        "duplicate_branch",
        "overlap_branch",
        "different_lines",
        "different_branches",
    ],
)
def test_coverage_rejects_unsupported_blended_gain(evidence: Path, breakage: str) -> None:
    def update(data):
        target = data["files"]["dashboard/pages/3_Scenario_Wizard.py"]
        if breakage == "line_only":
            data["meta"]["branch_coverage"] = False
        elif breakage == "summary_only":
            del target["missing_branches"]
        elif breakage == "missing_count":
            target["summary"]["missing_branches"] -= 1
        elif breakage == "duplicate_line":
            target["missing_lines"].append(target["missing_lines"][0])
        elif breakage == "overlap_line":
            target["missing_lines"][0] = target["executed_lines"][0]
        elif breakage == "duplicate_branch":
            target["missing_branches"].append(target["missing_branches"][0])
        elif breakage == "overlap_branch":
            target["missing_branches"][0] = target["executed_branches"][0]
        elif breakage == "different_lines":
            # Equal denominator sizes cannot establish equal coverage boundaries.
            target["executed_lines"][0] = 99999
        else:
            target["executed_branches"][0] = [99999, -99999]

    change_json(evidence, "recheck/coverage-merged.json", update)
    results = verifier.verify(evidence)
    assert results["coverage attribution"][0] == "FAIL"


def test_coverage_shortfall_does_not_pass_using_other_requirements(evidence: Path, capsys) -> None:
    # Restore the baseline's complete raw report: there is now zero measured gain.
    shutil.copyfile(
        evidence / "recheck/coverage-base.json", evidence / "recheck/coverage-merged.json"
    )
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "isolated measured reduction 0 units" in output
    assert "OVERALL FAIL" in output
