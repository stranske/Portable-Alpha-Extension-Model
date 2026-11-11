"""Validate repository-specific #2360 receipts; missing mandatory proof fails closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/evidence/issue-2360-closer-repair"
BASE = "10ac6839518576d4dc5199a24392eadfc388c713"
MERGED = "e8967edb2aa1bdbc1a68ea58ed241cf205915d03"
ASSESSED = "a2f4ef50403a0a90e8535e4daef5e6c691037bd2"
REPOSITORY = "stranske/Portable-Alpha-Extension-Model"
CONFIG_SHA256 = "b442a73dc1a9c83e895e6613a07f56d2604799da127ce14edee20adaf55d06d4"
PRODUCT_FILES_SHA256 = "acecd017d2b5bb4dd598d5bf791da0a526e936c01d16a6bd12ac6da445932499"
TARGET = "dashboard/pages/3_Scenario_Wizard.py"
WIZARD_SHA256 = "6a768b6a8bc41089b32cbf7eb32cd3d005e3db1934d5791cf3daee2710cea654"
EXPECTED_OUTCOME = (
    "Requirement-by-requirement PASS/FAIL/UNKNOWN; missing mandatory proof prevents OVERALL PASS"
)
RECORD_RECEIPTS = {
    "mutation-transcripts.md",
    "mutations.json",
    "recheck/revisions.json",
    "recheck/mutations/mutation-replay.json",
    "recheck/ci-run.json",
    "recheck/ci-pytest-job.json",
    "recheck/ci-pytest-excerpt.txt",
    "recheck/coverage-comparison.json",
    "ranking-provenance.json",
    "recheck/ranking-search.json",
}
ORIGINAL_CASES = (
    "test_progress_bar_marks_completed_current_and_future_steps",
    "test_capital_step_reports_allocation_status_and_balanced_chart[50.0-error]",
    "test_capital_step_reports_allocation_status_and_balanced_chart[60.0-success]",
    "test_capital_step_handles_uploaded_and_persisted_margin_schedules[True]",
    "test_capital_step_handles_uploaded_and_persisted_margin_schedules[False]",
    "test_sleeve_suggestor_explains_missing_or_empty_results[None]",
    "test_sleeve_suggestor_explains_missing_or_empty_results[empty]",
    "test_margin_schedule_loader_hides_invalid_schedule_errors",
)


def read_json(directory: Path, name: str):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def check_revisions(directory: Path) -> tuple[str, str]:
    record = read_json(directory, "recheck/revisions.json")
    # Pin API base revisions separately from the squash parent used for coverage.
    expected = {
        2372: {
            "base_sha": "e7c898dea47be8871dac1a29610be6679cdcbd57",
            "head_sha": "9bfb197c8ebb6f84f48b9beeac4e8cf6263d9847",
            "merge_sha": MERGED,
            "files": ["tests/test_wizard_render_coverage.py"],
        },
        2467: {
            "base_sha": "6abacb50a941ce1286e74afb5782d3c804427418",
            "head_sha": ASSESSED,
            "merge_sha": "bb8dd930e085b5c3bbc02b30b9f890eac537328c",
            "files": [
                "docs/evidence/issue-2360-closer-repair/README.md",
                "docs/evidence/issue-2360-closer-repair/ranking-provenance.json",
            ],
        },
    }
    prs = {pr["number"]: pr for pr in record["pull_requests"]}
    valid = (
        record["issue"]["number"] == 2360
        and record["issue"]["url"] == f"https://github.com/{REPOSITORY}/issues/2360"
        and len(record["pull_requests"]) == len(prs) == len(expected)
        and prs.keys() == expected.keys()
        and all(
            prs[number]["merged"] is True
            and prs[number]["state"] == "closed"
            and prs[number]["url"] == f"https://github.com/{REPOSITORY}/pull/{number}"
            and all(prs[number][key] == value for key, value in fields.items() if key != "files")
            and sorted(prs[number]["files"]) == fields["files"]
            for number, fields in expected.items()
        )
    )
    return ("PASS" if valid else "FAIL", "original delivery and assessed repair revisions")


def check_ci(directory: Path) -> tuple[str, str]:
    run = read_json(directory, "recheck/ci-run.json")
    jobs = read_json(directory, "recheck/ci-jobs.json")["jobs"]
    if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
        raise ValueError("CI jobs receipt requires a list of job objects")
    receipt = read_json(directory, "recheck/ci-pytest-job.json")
    pytest_job = receipt["job"]
    run_url = f"https://github.com/{REPOSITORY}/actions/runs/{run['id']}"
    api_run_url = f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{run['id']}"
    excerpt = (directory / "recheck/ci-pytest-excerpt.txt").read_text(encoding="utf-8")
    log = re.sub(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ", "", excerpt, flags=re.MULTILINE)
    checkouts = list(re.finditer(r"^[a-f0-9]{40}$", log, re.MULTILINE))
    commands = list(re.finditer(r'^pytest -p ci_test_growth "\$\{args\[@\]\}"$', log, re.MULTILINE))
    listed_jobs = [job for job in jobs if job.get("id") == pytest_job["id"]]

    def executed_pytest(job: dict) -> bool:
        steps = [
            step
            for step in job.get("steps") or []
            if step["name"] == "Pytest (unit tests with coverage)"
        ]
        return (
            len(steps) == 1
            and steps[0]["status"] == "completed"
            and steps[0]["conclusion"] == "success"
        )

    # A progress message or mixed passing/failing result is not successful pytest.
    summaries = list(
        re.finditer(
            r"=+[ \t]+([^\r\n]*\bin \d+(?:\.\d+)?s(?: \([^\r\n)]+\))?)[ \t]+=+[ \t]*$",
            log,
            re.MULTILINE,
        )
    )
    summary = summaries[-1].group(1) if summaries else ""
    valid = (
        run["name"] == "CI"
        and run["path"] == ".github/workflows/ci.yml"
        and run["head_sha"] == ASSESSED
        and run["status"] == "completed"
        and run["conclusion"] == "success"
        and run["html_url"] == run_url
        and receipt["source_url"] == f"{api_run_url}/jobs"
        and pytest_job["run_id"] == run["id"]
        and pytest_job["run_url"] == api_run_url
        and pytest_job["head_sha"] == ASSESSED
        and pytest_job["run_attempt"] == run["run_attempt"]
        and pytest_job["html_url"] == f"{run_url}/job/{pytest_job['id']}"
        and pytest_job["status"] == "completed"
        and pytest_job["conclusion"] == "success"
        and pytest_job["name"] == "Python CI / python 3.14"
        and executed_pytest(pytest_job)
        and bool(checkouts)
        and all(checkout.group() == ASSESSED for checkout in checkouts)
        and len(commands) == 1
        and checkouts[-1].end() < commands[0].start()
        and bool(summaries)
        and commands[0].end() < summaries[-1].start()
        and re.search(r"\b[1-9]\d* passed\b", summary) is not None
        and re.search(r"\b\d+ (?:failed|errors?)\b", summary) is None
        and len(listed_jobs) == 1
        and all(
            job["name"] == pytest_job["name"]
            and job["run_id"] == run["id"]
            and job["status"] == "completed"
            and job["conclusion"] == "success"
            and executed_pytest(job)
            for job in listed_jobs
        )
    )
    if valid:
        step = next(
            step
            for step in pytest_job["steps"]
            if step["name"] == "Pytest (unit tests with coverage)"
        )
        started = datetime.fromisoformat(step["started_at"])
        completed = datetime.fromisoformat(step["completed_at"])

        def logged_at(event: re.Match) -> datetime:
            line_number = log.count("\n", 0, event.start())
            timestamp = excerpt.splitlines()[line_number].partition(" ")[0]
            return datetime.fromisoformat(timestamp)

        invoked = logged_at(commands[0])
        finished = logged_at(summaries[-1])
        # REST step times have second precision; logs retain fractional seconds.
        valid = started <= completed and started <= invoked <= finished < completed + timedelta(
            seconds=1
        )
    return ("PASS" if valid else "FAIL", "exact-head executed pytest; Gate skip is insufficient")


def check_mutations(directory: Path) -> tuple[str, str]:
    record = read_json(directory, "recheck/mutations/mutation-replay.json")
    rows = record["results"]
    valid = (
        record["revision"] == MERGED
        and len(rows) == 8
        and {row["node"] for row in rows} == set(ORIGINAL_CASES)
        and record["wizard_sha256"] == WIZARD_SHA256
    )
    for row in rows:
        red_path = directory / "recheck/mutations" / row["red_output"]
        green_path = directory / "recheck/mutations" / row["green_output"]
        red, green = red_path.read_text(), green_path.read_text()
        node = f"tests/test_wizard_render_coverage.py::{row['node']}"
        command = shlex.split(row["command"])

        def single_case_result(output: str, outcome: str) -> bool:
            summaries = re.findall(
                r"^(?:=+ )?(.+?) in \d+(?:\.\d+)?s(?: \([^\n)]+\))?(?: =+)?$",
                output,
                re.MULTILINE,
            )
            return (
                bool(summaries)
                and re.fullmatch(rf"1 {outcome}(?:, [1-9]\d* warnings?)?", summaries[-1])
                is not None
            )

        valid = valid and (
            row["red_exit"] == 1
            and row["green_exit"] == 0
            and row["before"] != row["after"]
            and bool(row["name"])
            and bool(command)
            and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(command[0]).name) is not None
            and command[1:] == ["-B", "-m", "pytest", "-q", "-m", "not slow and not live_llm", node]
            and re.findall(r"^FAILED (\S+)(?: - .*)?$", red, re.MULTILINE) == [node]
            and single_case_result(red, "failed")
            and single_case_result(green, "passed")
            and "FAILED" not in green
            and row["restored_sha256"] == record["wizard_sha256"]
        )
    historical = read_json(directory, "mutations.json")["results"]
    later = [row["node"] for row in historical if row["node"] not in ORIGINAL_CASES]
    valid = valid and len(later) == len(set(later)) == 9
    return (
        "PASS" if valid else "FAIL",
        "eight original red/green proofs; nine later cases separate",
    )


def coverage_units(data: dict, path: str = TARGET) -> tuple[set, set, int]:
    """Validate the target's raw coverage units before trusting summary counts."""
    if data["meta"]["branch_coverage"] is not True:
        raise ValueError("statement-plus-branch attribution requires branch coverage")
    target = data["files"][path]
    summary = target["summary"]
    boundaries = []
    uncovered = 0
    for kind, denominator, covered in (
        ("lines", "num_statements", "covered_lines"),
        ("branches", "num_branches", "covered_branches"),
    ):
        executed_rows = target[f"executed_{kind}"]
        missing_rows = target[f"missing_{kind}"]
        executed = set(map(tuple, executed_rows)) if kind == "branches" else set(executed_rows)
        missing = set(map(tuple, missing_rows)) if kind == "branches" else set(missing_rows)
        if (
            len(executed) != len(executed_rows)
            or len(missing) != len(missing_rows)
            or executed & missing
            or len(executed) != summary[covered]
            or len(missing) != summary[f"missing_{kind}"]
            or len(executed | missing) != summary[denominator]
        ):
            raise ValueError(f"inconsistent raw coverage {kind} and summary")
        boundaries.append(executed | missing)
        uncovered += len(missing)
    return boundaries[0], boundaries[1], uncovered


def check_receipt(directory: Path, entry: dict, revision: str) -> tuple[dict, dict]:
    """Bind one paired-CI side to its checkout, command, environment and terminal result."""
    receipt = directory / entry["receipt_dir"]
    hashes = dict(
        line.split() for line in (receipt / "hashes.txt").read_text().splitlines() if line
    )
    command = shlex.split((receipt / "command.txt").read_text())
    # These are receipts for one completed replay, not a general pytest interface.
    # Matching commands can still both narrow selection or override coverage sources.
    expected_command = shlex.split(
        "pytest tests/ -m 'not live_llm' --junitxml=pytest-junit.xml --cov "
        "--cov-config=pyproject.toml --cov-branch --cov-report=json:coverage.json "
        "--cov-report=term-missing -n auto --dist=loadgroup"
    )
    suite = ET.parse(receipt / "pytest-junit.xml").getroot()
    suites = [suite] if suite.tag == "testsuite" else suite.findall("testsuite")
    if len(suites) != 1:
        raise ValueError("paired-CI receipt requires one complete pytest suite")
    suite = suites[0]
    counts = {key: int(suite.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}
    cases = {}
    for case in suite.findall("testcase"):
        identity = (case.attrib["classname"], case.attrib["name"])
        if identity in cases or case.find("failure") is not None or case.find("error") is not None:
            raise ValueError("duplicate or unsuccessful paired-CI testcase")
        cases[identity] = case.find("skipped") is not None
    if len(cases) != counts["tests"] or sum(cases.values()) != counts["skipped"]:
        raise ValueError("paired-CI testcase counts differ from suite summary")
    terminal = (receipt / "terminal-output.txt").read_text(encoding="utf-8")
    summaries = re.findall(
        r"=+[ \t]+([^\r\n]*\bin \d+(?:\.\d+)?s(?: \([^\r\n)]+\))?)[ \t]+=+[ \t]*$",
        terminal,
        re.MULTILINE,
    )
    # Use the final completed result; progress text or an earlier pass is insufficient.
    result = re.fullmatch(
        r"(\d+) passed(?:, (\d+) skipped)?(?:, \d+ warnings?)? in "
        r"\d+(?:\.\d+)?s(?: \([^\r\n)]+\))?",
        summaries[-1] if summaries else "",
    )
    if (
        result is None
        or int(result[1]) != counts["tests"] - counts["skipped"]
        or int(result[2] or 0) != counts["skipped"]
    ):
        raise ValueError("paired-CI terminal pytest result differs from successful JUnit suite")
    environment = read_json(receipt, "environment-check.json")
    job = re.fullmatch(
        rf"https://github\.com/{REPOSITORY}/actions/runs/(\d+)/job/\d+", entry["job_url"]
    )
    valid = (
        (receipt / "checkout-sha.txt").read_text().strip() == revision
        and hashes.get("pyproject.toml") == CONFIG_SHA256
        and (receipt / "exit-code.txt").read_text().strip() == str(entry["exit_code"]) == "0"
        and entry["coverage_record"] == f"{entry['receipt_dir']}/coverage.json"
        and command == expected_command
        and environment["identical"] is True
        and environment["mismatches"] == []
        and counts["tests"] > counts["skipped"]
        and counts["failures"] == counts["errors"] == 0
        and job is not None
    )
    if not valid:
        raise ValueError(f"paired-CI receipt does not bind {revision}")
    freeze = (receipt / "environment-freeze.txt").read_text().splitlines()
    platform = (receipt / "environment-platform.txt").read_text().splitlines()
    if not platform or not platform[0].strip():
        raise ValueError("paired-CI receipt is missing its Python execution context")
    return {
        "run_id": job.group(1),
        "command": command,
        "hashes": hashes,
        # Only the editable checkout path differs; the resolved package set must not.
        "environment": [
            line for line in freeze if not line.startswith("-e file:///tmp/issue2360-")
        ],
        "python": platform[0],
    }, cases


def check_paired_run(directory: Path, record: dict) -> bool:
    """Require successful jobs for both exact checkouts in the retained run."""
    run = read_json(directory, "recheck/paired-ci/run.json")
    binding = record["run"]
    run_url = f"https://github.com/{REPOSITORY}/actions/runs/{binding['id']}"
    if not (
        binding["conclusion"] == run["conclusion"] == "success"
        and binding["url"] == run["url"] == run_url
        and binding["head_sha"] == run["headSha"]
        and binding["attempt"] == run["attempt"]
        and binding["event"] == run["event"]
    ):
        return False
    for side, label, revision in (
        ("before", "configured-parent", BASE),
        ("after", "configured-merge", MERGED),
    ):
        matches = [job for job in run["jobs"] if job["url"] == record[side]["job_url"]]
        if len(matches) != 1:
            return False
        job = matches[0]
        required_steps = {
            f"Create isolated checkout at {revision}",
            "Execute historical configured test suite (bounded)",
            "Enforce complete successful suite",
        }
        if not (
            job["url"] == f"{run_url}/job/{job['databaseId']}"
            and job["name"] == f"Paired CI ({label})"
            and job["status"] == "completed"
            and job["conclusion"] == "success"
            and required_steps.issubset(
                step["name"]
                for step in job["steps"]
                if step["status"] == "completed" and step["conclusion"] == "success"
            )
        ):
            return False
    return True


def check_coverage(directory: Path) -> tuple[str, str]:
    record = read_json(directory, "recheck/coverage-comparison.json")
    if record["status"] == "UNKNOWN":
        return "UNKNOWN", record["limitation"]
    before, after = record["before"], record["after"]
    # The historical configured selection includes slow tests and excludes live_llm.
    # Matching focused selections establish comparability, not target attribution.
    if (
        record.get("scope") != "historical configured suite"
        or before["selection"] != "tests/ -m 'not live_llm'"
        or after["selection"] != "tests/ -m 'not live_llm'"
    ):
        return (
            "FAIL",
            "focused-only or unverified selection cannot prove configured-suite attribution",
        )
    valid = (
        record["status"] == "measured"
        and before["revision"] == BASE
        and after["revision"] == MERGED
        and before["exit_code"] == after["exit_code"] == 0
        and before["selection"] == after["selection"]
        and before["settings"] == after["settings"]
        and before["settings"]["branch"] is True
        and check_paired_run(directory, record)
    )
    parent_receipt, parent_cases = check_receipt(directory, before, BASE)
    merged_receipt, merged_cases = check_receipt(directory, after, MERGED)
    original_cases = {("tests.test_wizard_render_coverage", name): False for name in ORIGINAL_CASES}
    if parent_cases.keys() & original_cases.keys() or merged_cases != parent_cases | original_cases:
        return "FAIL", "executed suite delta must be exactly the eight successful original cases"
    # Both sides must come from one paired run with identical execution settings.
    valid = (
        valid
        and parent_receipt == merged_receipt
        and parent_receipt["run_id"] == str(record["run"]["id"])
    )
    measurements = []
    reports = []
    for entry in (before, after):
        data = read_json(directory, entry["coverage_record"])
        reports.append(data)
        measurements.append(coverage_units(data))
        target = data["files"][TARGET]
        recorded_counts = {
            "missing_lines": len(target["missing_lines"]),
            "missing_branches": len(target["missing_branches"]),
            "uncovered": measurements[-1][2],
        }
        if entry["target_units"] != recorded_counts:
            return "FAIL", "recorded target counts differ from raw coverage"
    first, last = measurements
    inventory = read_json(directory, "recheck/source-inventory.json")
    # Pin the complete product boundary from both historical Git trees. Two
    # reports can agree with each other while both omit the same untested file.
    if not (
        inventory["revisions"] == [BASE, MERGED]
        and inventory["sources"] == ["pa_core", "dashboard", "design-system"]
        and hashlib.sha256(json.dumps(inventory["files"], sort_keys=True).encode()).hexdigest()
        == PRODUCT_FILES_SHA256
        and all(report["files"].keys() == inventory["files"].keys() for report in reports)
    ):
        return "FAIL", "coverage inventory differs from the complete historical product boundary"
    # Raw unit boundaries must match for every measured file, not only the target.
    valid = (
        valid
        and first[:2] == last[:2]
        and reports[0]["files"].keys() == reports[1]["files"].keys()
        and all(
            coverage_units(reports[0], path)[:2] == coverage_units(reports[1], path)[:2]
            for path in reports[0]["files"]
        )
    )
    reduction = first[2] - last[2]
    if not valid:
        return "FAIL", "coverage revisions, execution, selection or unit boundaries differ"
    if reduction < 60:
        return "FAIL", f"isolated measured reduction {reduction} units; target about 60"
    return (
        "PASS",
        f"configured suite: {reduction} fewer statement-plus-branch units "
        f"({first[2]} to {last[2]} uncovered; "
        f"{len(first[0])} statements + {len(first[1])} branches; branch diagnostic)",
    )


def check_ranking(directory: Path) -> tuple[str, str]:
    record = read_json(directory, "ranking-provenance.json")
    provenance = record["historical_dispatch_provenance"]
    search = read_json(directory, "recheck/ranking-search.json")
    if (
        provenance["status"] == search["status"] == "UNKNOWN"
        and provenance["unavailable"]
        and search["sources_checked"]
        and search["matching_dispatch_receipt"] is None
        and search["matching_incident_mapping"] is None
    ):
        return (
            "PASS",
            "provenance explicitly UNKNOWN (permitted disclosure); no selection compliance claim",
        )
    return "FAIL", "matching dispatch and incident receipts must be independently inspected"


def check_contract(directory: Path) -> tuple[str, str]:
    record = read_json(directory, "recheck/contract.json")
    date.fromisoformat(record["checked_at"])
    attempts = record["stop_rule"]["attempts_per_step"]
    minutes = record["stop_rule"]["minutes"]
    narrative = (directory / "acceptance.md").read_text(encoding="utf-8")
    links = set(re.findall(r"\[[^\]\n]+\]\((?:\./)?([^\s)]+)\)", narrative))
    # The machine-readable contract cannot replace the durable, inspectable record.
    # Nonempty unrelated prose must not satisfy its date, revisions or receipt links.
    linked_record = (
        all(value in narrative for value in (record["checked_at"], BASE, MERGED, ASSESSED))
        and record["verification_command"] in narrative
        and RECORD_RECEIPTS <= links
    )
    valid = (
        record["verification_command"] == "python scripts/verify_issue_2360_evidence.py"
        and record["expected_outcome"] == EXPECTED_OUTCOME
        # JSON booleans compare as integers in Python, but are not retry/time limits.
        and type(attempts) is int
        and 0 < attempts <= 2
        and type(minutes) in (int, float)
        and 0 < minutes <= 45
        and linked_record
    )
    return (
        "PASS" if valid else "FAIL",
        "dated contract, runnable command, fail-closed outcome, finite stop rule; "
        "durable record binds revisions and receipt links",
    )


CHECKS = {
    "revisions": check_revisions,
    "CI": check_ci,
    "mutations": check_mutations,
    "coverage attribution": check_coverage,
    "ranking disclosure": check_ranking,
    "acceptance contract": check_contract,
}


def verify(directory: Path) -> dict[str, tuple[str, str]]:
    results = {}
    for name, check in CHECKS.items():
        try:
            results[name] = check(directory)
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
            results[name] = ("FAIL", f"missing or invalid evidence: {error}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    results = verify(args.evidence_dir)
    for name, (status, detail) in results.items():
        print(f"{status}: {name}: {detail}")
    passed = all(status == "PASS" for status, _detail in results.values())
    print("OVERALL PASS" if passed else "OVERALL FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
