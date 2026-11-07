"""Validate repository-specific #2360 receipts; missing mandatory proof fails closed."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/evidence/issue-2360-closer-repair"
BASE = "10ac6839518576d4dc5199a24392eadfc388c713"
MERGED = "e8967edb2aa1bdbc1a68ea58ed241cf205915d03"
ASSESSED = "a2f4ef50403a0a90e8535e4daef5e6c691037bd2"
WIZARD_SHA256 = "6a768b6a8bc41089b32cbf7eb32cd3d005e3db1934d5791cf3daee2710cea654"
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
    prs = {pr["number"]: pr for pr in record["pull_requests"]}
    valid = (
        prs[2372]["merge_sha"] == MERGED
        and prs[2467]["head_sha"] == ASSESSED
        and all(prs[n]["merged"] for n in (2372, 2467))
        and prs[2372]["files"] == ["tests/test_wizard_render_coverage.py"]
    )
    return ("PASS" if valid else "FAIL", "original delivery and assessed repair revisions")


def check_ci(directory: Path) -> tuple[str, str]:
    run = read_json(directory, "recheck/ci-run.json")
    jobs = read_json(directory, "recheck/ci-jobs.json")["jobs"]
    excerpt = (directory / "recheck/ci-pytest-excerpt.txt").read_text()
    valid = (
        run["head_sha"] == ASSESSED
        and run["status"] == "completed"
        and run["conclusion"] == "success"
        and run["html_url"].endswith(f"/actions/runs/{run['id']}")
        and ASSESSED in excerpt
        and 'pytest -p ci_test_growth "${args[@]}"' in excerpt
        and re.search(r"\b\d+ passed\b", excerpt) is not None
        and any(
            job["run_id"] == run["id"]
            and job["status"] == "completed"
            and job["conclusion"] == "success"
            and any(
                step["name"] == "Pytest (unit tests with coverage)"
                and step["status"] == "completed"
                and step["conclusion"] == "success"
                for step in job.get("steps") or []
            )
            for job in jobs
        )
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
        valid = valid and (
            row["red_exit"] == 1
            and row["green_exit"] == 0
            and row["before"] != row["after"]
            and bool(row["name"])
            and node in row["command"]
            and "not slow" in row["command"]
            and f"FAILED {node}" in red
            and "1 failed" in red
            and "1 passed" in green
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


def check_coverage(directory: Path) -> tuple[str, str]:
    record = read_json(directory, "recheck/coverage-comparison.json")
    if record["status"] == "UNKNOWN":
        return "UNKNOWN", record["limitation"]
    before, after = record["before"], record["after"]
    valid = (
        before["revision"] == BASE
        and after["revision"] == MERGED
        and before["exit_code"] == after["exit_code"] == 0
        and before["selection"] == after["selection"]
        and before["settings"] == after["settings"]
        and before["settings"]["branch"] is True
    )
    summaries = []
    for entry in (before, after):
        data = read_json(directory, entry["coverage_record"])
        summaries.append(data["files"]["dashboard/pages/3_Scenario_Wizard.py"]["summary"])
    first, last = summaries
    valid = valid and all(first[key] == last[key] for key in ("num_statements", "num_branches"))
    reduction = sum(first[key] - last[key] for key in ("missing_lines", "missing_branches"))
    if not valid:
        return "FAIL", "coverage revisions, execution, selection or denominators differ"
    if reduction < 60:
        return "FAIL", f"isolated measured reduction {reduction} units; target about 60"
    return (
        "PASS",
        f"isolated wizard selection: {reduction} fewer statement-plus-branch units; not Gate",
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
    valid = (
        record["verification_command"] == "python scripts/verify_issue_2360_evidence.py"
        and 0 < record["stop_rule"]["attempts_per_step"] <= 2
        and 0 < record["stop_rule"]["minutes"] <= 45
        and (directory / "acceptance.md").is_file()
    )
    return ("PASS" if valid else "FAIL", "dated contract, runnable command, finite stop rule")


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
        except (OSError, ValueError, KeyError, TypeError) as error:
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
