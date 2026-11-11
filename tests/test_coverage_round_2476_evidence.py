"""Guard the issue #2476 seven-target evidence against silent drift.

The measured reduction (209 -> 2 uncovered statements, 123 -> 4 uncovered
branches at ``54219bd7``) and the three documented remaining gaps are recorded
in ``docs/evidence/issue-2476/``. This test fails if the JSON measurements,
the README, the production facade fix, or the tests covering the remaining
gaps stop agreeing with each other, without re-running the Gate coverage
command.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "issue-2476"
MEASURED_REVISION = "54219bd739793fa754c3fec319bb2c6b8f8dd898"
MAIN_REVISION = "5be77e15c925f5354ab67e2a1749fe6187d2038c"
MERGE_REVISION = "ab96b5b495547fcad317da7c135d9b082aa0e150"


def _measurements() -> dict:
    return json.loads((EVIDENCE / "gate-measurements.json").read_text(encoding="utf-8"))


def _test_names(path: Path) -> set[str]:
    return set(re.findall(r"^def (test_\w+)\(", path.read_text(encoding="utf-8"), re.M))


def test_seven_target_reduction_arithmetic_is_consistent() -> None:
    data = _measurements()
    assert data["measured_revision"] == MEASURED_REVISION
    assert data["round_baseline_uncovered_statements"] == 209
    assert data["uncovered_statement_reduction"] == 207
    assert data["remaining_uncovered_statements"] == 2
    assert (
        data["round_baseline_uncovered_statements"] - data["uncovered_statement_reduction"]
        == data["remaining_uncovered_statements"]
    )
    assert data["round_baseline_uncovered_branches"] == 123
    assert data["uncovered_branch_reduction"] == 119
    assert data["remaining_uncovered_branches"] == 4
    assert (
        data["round_baseline_uncovered_branches"] - data["uncovered_branch_reduction"]
        == data["remaining_uncovered_branches"]
    )
    assert len(data["seven_targets_in_dispatch_order"]) == 7

    per_target = data["statement_attribution"]["per_target_round_baseline_to_delivered"]
    assert set(per_target) == set(data["seven_targets_in_dispatch_order"])
    assert sum(v["statements"] for v in per_target.values()) == 207
    assert sum(v["branches"] for v in per_target.values()) == 119

    chronological = data["statement_attribution"]["chronological"]
    assert sum(v["statements"] for v in chronological.values()) == 207
    assert sum(v["branches"] for v in chronological.values()) == 119


def test_gate_coverage_delivered_snapshot_matches_measurements() -> None:
    coverage = json.loads((EVIDENCE / "gate-coverage.json").read_text(encoding="utf-8"))
    measurements = _measurements()
    delivered = coverage["snapshots"]["delivered"]
    assert delivered["revision"] == MEASURED_REVISION
    assert coverage["ranking"]["dispatch_total"] == 209

    statements = delivered["statements"]
    assert statements["selected_missing_statements"] == 2
    assert sum(t["missing_statements"] for t in statements["targets"].values()) == 2
    assert statements["targets"]["pa_core/facade.py"]["missing_statement_lines"] == [750, 756]

    branches = delivered["branches"]
    assert branches["selected_missing_statements"] == 2
    assert branches["selected_missing_branches"] == 4
    assert sum(t["missing_branches"] for t in branches["targets"].values()) == 4
    exceptions = measurements["remaining_exceptions"]
    for target, entry in exceptions.items():
        if not isinstance(entry, dict):
            continue
        snapshot = branches["targets"][target]
        assert snapshot["missing_statement_lines"] == entry["missing_statements"]
        assert snapshot["missing_branch_arcs"] == entry["missing_branch_arcs"]

    per_target = measurements["statement_attribution"]["per_target_round_baseline_to_delivered"]
    assert coverage["statement_reductions"]["round_baseline_to_delivered"] == 207
    assert coverage["statement_reductions"]["per_target_round_baseline_to_delivered"] == {
        target: values["statements"] for target, values in per_target.items()
    }
    assert coverage["branch_reductions"]["round_baseline_to_delivered"] == 119
    assert coverage["branch_reductions"]["per_target_round_baseline_to_delivered"] == {
        target: values["branches"] for target, values in per_target.items()
    }


def test_chronological_reductions_sum_to_the_delivered_totals() -> None:
    coverage = json.loads((EVIDENCE / "gate-coverage.json").read_text(encoding="utf-8"))
    measurements = _measurements()
    chronological = measurements["statement_attribution"]["chronological"]
    assert len(chronological) == 3

    statement_steps = coverage["statement_reductions"]
    assert (
        statement_steps["round_baseline_to_after_2384"]
        + statement_steps["after_2384_to_before_this_work"]
        + statement_steps["this_work"]
        == statement_steps["round_baseline_to_delivered"]
        == sum(values["statements"] for values in chronological.values())
    )
    branch_steps = coverage["branch_reductions"]
    assert (
        branch_steps["round_baseline_to_after_2384"]
        + branch_steps["after_2384_to_before_this_work"]
        + branch_steps["this_work"]
        == branch_steps["round_baseline_to_delivered"]
        == sum(values["branches"] for values in chronological.values())
    )

    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    for row in (
        "| round baseline → after merged PR #2384 | 58 | 14 |",
        "| after #2384 → before this work (main up to the #2476 bootstrap) | 131 | 79 |",
        "| before this work → delivered (this PR) | 18 | 26 |",
    ):
        assert row in readme


def test_remaining_gaps_match_readme_and_covering_tests() -> None:
    exceptions = _measurements()["remaining_exceptions"]
    assert exceptions["pa_core/facade.py"]["missing_statements"] == [750, 756]
    assert exceptions["pa_core/facade.py"]["missing_branch_arcs"] == [[749, 750], [755, 756]]
    assert exceptions["pa_core/pa.py"]["missing_branch_arcs"] == [[349, -120]]
    assert exceptions["dashboard/pages/4_Results.py"]["missing_branch_arcs"] == [[178, 180]]
    for entry in exceptions.values():
        if isinstance(entry, dict):
            assert entry["reason"]

    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    assert MEASURED_REVISION in readme
    assert "209 → 2" in readme
    assert "123 → 4" in readme
    assert "statements 750 and 756" in readme
    assert "arc 349→exit" in readme
    assert "arc 178→180" in readme


def _hosted_main_receipts() -> dict:
    path = EVIDENCE / "continuation-20261111" / "hosted-main" / "seven-targets.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_receipts_bind_remaining_gaps_to_measured_revisions() -> None:
    receipts = _hosted_main_receipts()
    assert set(receipts) == {"main-gate", "main-branch", "merge-gate", "merge-branch"}
    for label, receipt in receipts.items():
        expected = MAIN_REVISION if label.startswith("main-") else MERGE_REVISION
        assert receipt["revision"] == expected
        assert receipt["exit_code"] == 0
        assert receipt["branch_coverage"] is label.endswith("-branch")
        revision_file = EVIDENCE / "continuation-20261111" / "hosted-main" / label / "revision.txt"
        assert revision_file.read_text(encoding="utf-8").strip() == expected

    merge = receipts["merge-branch"]["targets"]
    gaps = {
        target: (entry["uncovered_statement_lines"], entry["uncovered_branch_arcs"])
        for target, entry in merge.items()
        if entry["uncovered_statement_count"] or entry["uncovered_branch_count"]
    }
    assert gaps == {
        "pa_core/facade.py": ([751, 757], [[750, 751], [756, 757]]),
        "pa_core/pa.py": ([], [[349, -120]]),
        "dashboard/pages/4_Results.py": ([], [[178, 180]]),
    }
    assert receipts["merge-gate"]["targets"]["pa_core/facade.py"]["uncovered_statement_lines"] == [
        751,
        757,
    ]
    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    assert "statements\n751 and 757 with arcs 750→751 and 756→757" in readme


def _signed(value: int) -> str:
    return f"−{-value}" if value < 0 else str(value)


def test_readme_per_target_deltas_match_current_receipts() -> None:
    receipts = _hosted_main_receipts()
    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    current = readme.split("## Current receipts", 1)[1].split("\n## ", 1)[0]
    statement_total = branch_total = 0
    for target in _measurements()["seven_targets_in_dispatch_order"]:
        statements = (
            receipts["merge-gate"]["targets"][target]["uncovered_statement_count"]
            - receipts["main-gate"]["targets"][target]["uncovered_statement_count"]
        )
        branches = (
            receipts["merge-branch"]["targets"][target]["uncovered_branch_count"]
            - receipts["main-branch"]["targets"][target]["uncovered_branch_count"]
        )
        statement_total += statements
        branch_total += branches
        assert f"| `{target}` | {_signed(statements)} | {_signed(branches)} |" in current
    assert (statement_total, branch_total) == (-18, -26)
    assert "| **Total (this PR)** | **−18 (20 → 2)** | **−26 (30 → 4)** |" in current


def test_current_main_reconciles_with_historical_before_this_work_snapshot() -> None:
    receipts = _hosted_main_receipts()
    coverage = json.loads((EVIDENCE / "gate-coverage.json").read_text(encoding="utf-8"))
    historical = coverage["snapshots"]["before_this_work"]
    for target, entry in historical["statements"]["targets"].items():
        main_gate = receipts["main-gate"]["targets"][target]
        assert main_gate["uncovered_statement_count"] == entry["missing_statements"], target
    for target, entry in historical["branches"]["targets"].items():
        main_branch = receipts["main-branch"]["targets"][target]
        assert main_branch["uncovered_branch_count"] == entry["missing_branches"], target
    # The PR step reconciles; the earlier steps remain historical-policy figures.
    assert coverage["statement_reductions"]["this_work"] == 18
    assert coverage["branch_reductions"]["this_work"] == 26
    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    assert "| round baseline → after merged PR #2384 | 58 | 14 | historical test policy |" in readme
    assert (
        "| after #2384 → before this work (intervening changes) | 131 | 79 | "
        "historical test policy |" in readme
    )
    assert "| main → prospective merge (this PR) | 18 | 26 | current hosted receipts |" in readme
    assert "matched round\nbaseline therefore remains unmeasured" in readme
    anchor = (
        EVIDENCE / "continuation-20261111" / "hosted-anchor" / "failed-install.txt"
    ).read_text(encoding="utf-8")
    assert "bf424ec6ff58bb88b6431acf7809d42b2c59f95d" in anchor
    assert anchor.count("coverage==7.17.2 and coverage==7.16.2") == 2


def test_calibration_copy_mutation_is_labelled_identity_only() -> None:
    calibration = EVIDENCE / "calibration-20261111"
    assert "is **identity-only**" in (calibration / "README.md").read_text(encoding="utf-8")
    transcripts = (calibration / "mutation-transcripts.md").read_text(encoding="utf-8")
    record = transcripts.split("## calibration-frame\n", 1)[1]
    assert record.lstrip().startswith("Scope: **identity-only**.")


def test_facade_mixed_series_dict_fix_and_its_test_are_present() -> None:
    facade_source = (ROOT / "pa_core" / "facade.py").read_text(encoding="utf-8")
    assert "item.to_dict() if isinstance(item, pd.Series) else item" in facade_source

    facade_tests = ROOT / "tests" / "test_coverage_round_2476_facade.py"
    assert "test_serialize_agent_semantics_from_series_and_dict_list" in _test_names(facade_tests)
    test_source = facade_tests.read_text(encoding="utf-8")
    assert 'pd.Series({"Agent": "A"})' in test_source
    assert '{"Agent": "B"}' in test_source
    # The strengthened records replaced the ineffective scalar/column cases.
    assert "np.int64(1)" in test_source
    assert "np.nan" in test_source
    assert 'float("inf")' in test_source


def test_mutation_transcripts_still_record_41_red_green_pairs() -> None:
    text = (EVIDENCE / "mutation-transcripts.md").read_text(encoding="utf-8")
    assert text.count("- Red exit: 1") == 41
    assert text.count("- Green exit: 0") == 41
    assert "facade.py:143" in text
    assert "AttributeError: 'dict' object has no attribute 'dtype'" in text


def test_mutation_transcripts_each_record_restores_production_sha256() -> None:
    text = (EVIDENCE / "mutation-transcripts.md").read_text(encoding="utf-8")
    records = [section for section in text.split("\n## `") if "- Red exit: 1" in section]
    assert len(records) == 41
    for record in records:
        title = record.split("`", 1)[0]
        hashes = dict(
            re.findall(r"- sha256 (before|mutated|restored): `([0-9a-f]{64})`", record),
        )
        assert set(hashes) == {"before", "mutated", "restored"}, title
        assert hashes["restored"] == hashes["before"], title
        assert hashes["mutated"] != hashes["before"], title
        assert "(matches)" in record, title
        assert "- File: `" in record, title
        assert "- Green exit: 0" in record, title
        assert "python -m pytest" in record, title
    # The two ineffective earlier records were replaced, not kept alongside.
    assert text.count("- Record: replaced:") == 3
    assert "test_serialize_agent_semantics_from_column_mapping" in text
    assert "pa_core/facade.py:143" in text


def test_historical_coverage_role_acknowledges_later_test_changes() -> None:
    coverage = json.loads((EVIDENCE / "gate-coverage.json").read_text(encoding="utf-8"))
    role = coverage["snapshots"]["delivered"]["role"]
    assert "added test cases" in role
    assert "coverage was not remeasured" in role
    readme = (EVIDENCE / "README.md").read_text(encoding="utf-8")
    assert "Full-suite coverage for the delivered head remains unverified" not in readme
    current, historical = readme.split(
        "## Acceptance records (2026-11-10) — historical snapshot", 1
    )
    assert "## Current receipts (revision-bound, 2026-11-11)" in current
    assert MAIN_REVISION in current and MERGE_REVISION in current
    assert MEASURED_REVISION not in current
    assert MEASURED_REVISION in historical


def test_baseline_timeout_summary_preserves_reported_failure_and_unknowns() -> None:
    transcript = (EVIDENCE / "review-20261111/full-baseline-timeout.txt").read_text()
    assert any("F" in line and "[ 38%]" in line for line in transcript.splitlines())
    summary = (EVIDENCE / "review-20261111.md").read_text()
    assert "at least one test failure at 38%" in summary
    assert "failed test's identity and the timeout cause remain **unknown**" in summary
    assert "98% progress without a reported" not in summary
