"""Acceptance must reject stale, skipped, or incomplete historical receipts."""

import json
import shutil
import xml.etree.ElementTree as ET
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
    assert verifier.check_coverage(verifier.DEFAULT) == (
        "PASS",
        "configured suite: 73 fewer statement-plus-branch units "
        "(463 to 390 uncovered; 1050 statements + 284 branches; branch diagnostic)",
    )
    assert verifier.check_contract(verifier.DEFAULT)[0] == "PASS"


@pytest.mark.parametrize("outcome", [None, "", "OVERALL PASS", "PASS even if CI is missing"])
def test_contract_requires_recorded_fail_closed_outcome(evidence: Path, outcome, capsys) -> None:
    def update(record):
        if outcome is None:
            record.pop("expected_outcome")
        else:
            record["expected_outcome"] = outcome

    change_json(evidence, "recheck/contract.json", update)
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "FAIL: acceptance contract:" in output
    assert "OVERALL FAIL" in output
    assert "OVERALL PASS" not in output


@pytest.mark.parametrize(
    ("field", "value"), [("attempts_per_step", True), ("attempts_per_step", 1.5), ("minutes", True)]
)
def test_contract_requires_numeric_time_and_integer_attempts(
    evidence: Path, field: str, value, capsys
) -> None:
    change_json(
        evidence, "recheck/contract.json", lambda record: record["stop_rule"].update({field: value})
    )
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "FAIL: acceptance contract:" in output
    assert "OVERALL FAIL" in output
    assert "OVERALL PASS" not in output


@pytest.mark.parametrize(("attempts", "minutes"), [(1, 1), (2, 45), (2, 1.5)])
def test_contract_accepts_bounded_numeric_limits(
    evidence: Path, attempts: int, minutes: int | float
) -> None:
    change_json(
        evidence,
        "recheck/contract.json",
        lambda record: record.update(stop_rule={"attempts_per_step": attempts, "minutes": minutes}),
    )
    assert verifier.check_contract(evidence)[0] == "PASS"


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


@pytest.mark.parametrize(
    "summary",
    [
        "========== 0 passed in 0.01s ==========",
        "========== 1 passed, 1 failed in 0.01s ==========",
        "========== 1 failed, 1 passed in 0.01s ==========",
        "========== 1 passed, 1 error in 0.01s ==========",
        "========== 1 passed, 2 errors in 0.01s ==========",
        "Progress report: 1 passed",
    ],
)
def test_ci_requires_successful_terminal_pytest_result(
    evidence: Path, summary: str, capsys
) -> None:
    path = evidence / "recheck/ci-pytest-excerpt.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [summary if "1969 passed" in line else line for line in lines]
    path.write_text("\n".join(lines), encoding="utf-8")
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


PAIRED = "recheck/paired-ci"
PARENT_COVERAGE = f"{PAIRED}/configured-parent/coverage.json"
MERGE_COVERAGE = f"{PAIRED}/configured-merge/coverage.json"


@pytest.fixture
def configured_evidence(evidence: Path) -> Path:
    # The committed paired-CI receipts from run 42887512783, unchanged.
    return evidence


@pytest.mark.parametrize("breakage", [None, "later_revision", "selection", "denominator"])
def test_coverage_requires_isolated_comparable_execution(
    configured_evidence: Path, breakage: str
) -> None:
    evidence = configured_evidence
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
            MERGE_COVERAGE,
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
def test_coverage_rejects_unsupported_blended_gain(
    configured_evidence: Path, breakage: str
) -> None:
    evidence = configured_evidence

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

    change_json(evidence, MERGE_COVERAGE, update)
    results = verifier.verify(evidence)
    assert results["coverage attribution"][0] == "FAIL"


def test_coverage_shortfall_does_not_pass_using_other_requirements(
    configured_evidence: Path, capsys
) -> None:
    evidence = configured_evidence
    # Restore the baseline's complete raw report: there is now zero measured gain.
    shutil.copyfile(evidence / PARENT_COVERAGE, evidence / MERGE_COVERAGE)
    change_json(
        evidence,
        "recheck/coverage-comparison.json",
        lambda record: record["after"].update(target_units=record["before"]["target_units"]),
    )
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "isolated measured reduction 0 units" in output
    assert "OVERALL FAIL" in output


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("field", ["missing_lines", "missing_branches", "uncovered", None])
def test_recorded_target_counts_must_match_raw_coverage(
    evidence: Path, side: str, field: str | None, capsys
) -> None:
    def update(record):
        if field is None:
            del record[side]["target_units"]
        else:
            record[side]["target_units"][field] += 1

    change_json(evidence, "recheck/coverage-comparison.json", update)
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "FAIL: coverage attribution:" in output
    assert "OVERALL FAIL" in output
    assert "OVERALL PASS" not in output


@pytest.mark.parametrize("relabel_scope", [False, True])
def test_current_focused_receipt_cannot_satisfy_mandatory_attribution(
    evidence: Path, monkeypatch: pytest.MonkeyPatch, capsys, relabel_scope: bool
) -> None:
    # Keep this real receipt as-is: it measured 73 units only in the wizard slice.
    record = verifier.read_json(evidence, "recheck/coverage-comparison.json")
    focused = record.get("supplemental_focused", record)
    if relabel_scope:
        focused["scope"] = "historical configured suite"
    (evidence / "recheck/coverage-comparison.json").write_text(json.dumps(focused))
    before = verifier.coverage_units(
        verifier.read_json(evidence, focused["before"]["coverage_record"])
    )
    after = verifier.coverage_units(
        verifier.read_json(evidence, focused["after"]["coverage_record"])
    )
    assert before[2] - after[2] == 73
    monkeypatch.setattr(verifier, "CHECKS", {"coverage attribution": verifier.check_coverage})
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "focused-only" in output
    assert "OVERALL FAIL" in output
    assert "OVERALL PASS" not in output


def _replace_text(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


@pytest.mark.parametrize("label", ["configured-parent", "configured-merge"])
@pytest.mark.parametrize(
    "breakage", ["missing", "truncated", "failed", "wrong_passed", "wrong_skipped", "later_error"]
)
def test_configured_terminal_result_must_agree_with_junit(
    evidence: Path, label: str, breakage: str, capsys
) -> None:
    path = evidence / PAIRED / label / "terminal-output.txt"
    passed = 1716 if label == "configured-parent" else 1724
    if breakage == "missing":
        path.unlink()
    elif breakage == "truncated":
        path.write_text(f"{passed} passed so far; waiting for remaining tests\n")
    elif breakage == "failed":
        _replace_text(path, f"{passed} passed,", f"1 failed, {passed} passed,")
    elif breakage == "wrong_passed":
        _replace_text(path, f"{passed} passed,", f"{passed - 1} passed,")
    elif breakage == "wrong_skipped":
        _replace_text(path, "2 skipped,", "1 skipped,")
    else:
        # Earlier success cannot hide the last, unsuccessful terminal result.
        path.write_text(path.read_text() + "=== 1 error in 0.10s ===\n")
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "FAIL: coverage attribution:" in output
    assert "OVERALL FAIL" in output
    assert "OVERALL PASS" not in output


@pytest.mark.parametrize(
    "extra_args",
    [
        " -k wizard",
        " --lf",
        " --deselect=tests/test_wizard_render_coverage.py",
        " -m 'not slow and not live_llm'",
        " --cov=dashboard",
    ],
)
def test_matching_filtered_commands_cannot_prove_configured_suite(
    evidence: Path, extra_args: str, capsys
) -> None:
    for label in ("configured-parent", "configured-merge"):
        command = evidence / PAIRED / label / "command.txt"
        command.write_text(command.read_text().strip() + extra_args)
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "FAIL: coverage attribution:" in output
    assert "OVERALL FAIL" in output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conclusion", "failure"),
        ("headSha", "0" * 40),
        ("attempt", 2),
        ("event", "workflow_dispatch"),
        ("url", "https://github.com/unrelated/repository/actions/runs/42887512783"),
    ],
)
def test_coverage_binds_comparison_to_retained_run(evidence: Path, field: str, value) -> None:
    change_json(evidence, f"{PAIRED}/run.json", lambda run: run.update({field: value}))
    assert verifier.verify(evidence)["coverage attribution"][0] == "FAIL"


@pytest.mark.parametrize(
    "breakage",
    [
        "missing_job",
        "job_url",
        "job_id",
        "job_name",
        "job_status",
        "job_failure",
        "wrong_checkout",
        "skipped_execution",
        "missing_enforcement",
    ],
)
def test_coverage_requires_successful_historical_job(evidence: Path, breakage: str) -> None:
    def update(run):
        job = run["jobs"][1]
        if breakage == "missing_job":
            run["jobs"].pop(1)
        elif breakage == "job_url":
            job["url"] += "0"
        elif breakage == "job_id":
            job["databaseId"] += 1
        elif breakage == "job_name":
            job["name"] = "Paired CI (configured-parent)"
        elif breakage == "job_status":
            job["status"] = "in_progress"
        elif breakage == "job_failure":
            job["conclusion"] = "failure"
        elif breakage == "wrong_checkout":
            job["steps"][3]["name"] = f"Create isolated checkout at {verifier.BASE}"
        elif breakage == "skipped_execution":
            job["steps"][6]["conclusion"] = "skipped"
        else:
            job["steps"].pop(8)

    change_json(evidence, f"{PAIRED}/run.json", update)
    assert verifier.verify(evidence)["coverage attribution"][0] == "FAIL"


@pytest.mark.parametrize(
    "breakage",
    [
        "checkout_sha",
        "config_hash",
        "split_marker",
        "no_branch",
        "different_command",
        "environment_mismatch",
        "different_package",
        "nonzero_exit",
        "failing_junit",
        "empty_junit",
        "foreign_job",
        "other_run",
        "coverage_outside_receipt",
        "other_file_boundary",
    ],
)
def test_coverage_requires_bound_paired_ci_receipts(
    configured_evidence: Path, breakage: str
) -> None:
    evidence = configured_evidence
    parent = evidence / PAIRED / "configured-parent"
    merge = evidence / PAIRED / "configured-merge"
    comparison = "recheck/coverage-comparison.json"
    if breakage == "checkout_sha":
        (merge / "checkout-sha.txt").write_text(verifier.BASE + "\n")
    elif breakage == "config_hash":
        _replace_text(merge / "hashes.txt", verifier.CONFIG_SHA256, "0" * 64)
    elif breakage == "split_marker":
        for side in (parent, merge):
            # Run 42874673712's scalar expansion gave -m the tokens "'not" and "live_llm'".
            _replace_text(side / "command.txt", "-m not\\ live_llm", "-m \\'not live_llm\\'")
    elif breakage == "no_branch":
        for side in (parent, merge):
            _replace_text(side / "command.txt", "--cov-branch ", "")
    elif breakage == "different_command":
        _replace_text(merge / "command.txt", "-n auto", "-n 2")
    elif breakage == "environment_mismatch":
        change_json(
            evidence,
            f"{PAIRED}/configured-merge/environment-check.json",
            lambda record: record.update(identical=False, mismatches=["pytest"]),
        )
    elif breakage == "different_package":
        _replace_text(merge / "environment-freeze.txt", "pytest==9.1.1", "pytest==9.2.0")
    elif breakage == "nonzero_exit":
        (merge / "exit-code.txt").write_text("1\n")
    elif breakage == "failing_junit":
        _replace_text(merge / "pytest-junit.xml", 'failures="0"', 'failures="1"')
    elif breakage == "empty_junit":
        _replace_text(merge / "pytest-junit.xml", 'tests="1726"', 'tests="2"')
    elif breakage == "foreign_job":
        change_json(
            evidence,
            comparison,
            lambda record: record["after"].update(
                job_url="https://github.com/unrelated/repository/actions/runs/1/job/2"
            ),
        )
    elif breakage == "other_run":
        change_json(evidence, comparison, lambda record: record["run"].update(id=1))
    elif breakage == "coverage_outside_receipt":
        change_json(
            evidence,
            comparison,
            lambda record: record["after"].update(coverage_record="recheck/coverage-merged.json"),
        )
    else:

        def update(data):
            data["files"]["pa_core/cli.py"]["executed_lines"][0] = 99999

        change_json(evidence, MERGE_COVERAGE, update)
    assert verifier.verify(evidence)["coverage attribution"][0] == "FAIL"


@pytest.mark.parametrize(
    "breakage",
    [
        "missing_original",
        "later_case",
        "skipped_original",
        "failed_original",
        "duplicate_original",
        "missing_shared",
        "changed_shared",
        "summary_only",
        "original_in_parent",
        "malformed_report",
    ],
)
def test_configured_attribution_requires_exact_executed_case_delta(
    evidence: Path, breakage: str, capsys
) -> None:
    path = evidence / PAIRED / "configured-merge" / "pytest-junit.xml"
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    original = next(case for case in suite if case.get("name") == verifier.ORIGINAL_CASES[0])
    shared = next(case for case in suite if case.get("classname") != original.get("classname"))
    if breakage == "missing_original":
        suite.remove(original)
        suite.set("tests", str(int(suite.get("tests")) - 1))
    elif breakage == "later_case":
        original.set("name", "test_added_after_original_merge")
    elif breakage == "skipped_original":
        ET.SubElement(original, "skipped")
        suite.set("skipped", str(int(suite.get("skipped")) + 1))
    elif breakage == "failed_original":
        # A contradictory success summary must not hide a failing testcase.
        ET.SubElement(original, "failure")
    elif breakage == "duplicate_original":
        suite.append(ET.fromstring(ET.tostring(original)))
        suite.set("tests", str(int(suite.get("tests")) + 1))
    elif breakage == "missing_shared":
        suite.remove(shared)
        suite.set("tests", str(int(suite.get("tests")) - 1))
    elif breakage == "changed_shared":
        shared.set("name", "unrelated_replacement_case")
    elif breakage == "summary_only":
        suite.remove(original)  # Leave the reported total untouched.
    elif breakage == "original_in_parent":
        parent_path = evidence / PAIRED / "configured-parent" / "pytest-junit.xml"
        parent_tree = ET.parse(parent_path)
        parent_suite = parent_tree.getroot().find("testsuite")
        parent_suite.append(original)
        parent_suite.set("tests", str(int(parent_suite.get("tests")) + 1))
        parent_tree.write(parent_path)
    tree.write(path)
    if breakage == "malformed_report":
        path.write_text("<testsuites>")
    assert verifier.main(["--evidence-dir", str(evidence)]) == 1
    output = capsys.readouterr().out
    assert "FAIL: coverage attribution:" in output
    assert "OVERALL FAIL" in output
    assert "OVERALL PASS" not in output
