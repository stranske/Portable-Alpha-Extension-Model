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


def test_current_production_lines_match_documented_remaining_gaps() -> None:
    facade_source = (ROOT / "pa_core" / "facade.py").read_text(encoding="utf-8")
    export_source = facade_source.split("\ndef export(", 1)[1]
    assert (
        "if isinstance(agent_semantics_val, pd.DataFrame) and not agent_semantics_val.empty:\n"
        "            serialized_agent_semantics = _records_to_builtin(" in export_source
    )
    assert (
        "if isinstance(serialized_agent_semantics, tuple):\n"
        "                serialized_agent_semantics = list(" in export_source
    )

    pa_lines = (ROOT / "pa_core" / "pa.py").read_text(encoding="utf-8").splitlines()
    assert pa_lines[348].strip() == 'elif args.registry_cmd == "get":'

    results_path = ROOT / "dashboard" / "pages" / "4_Results.py"
    results_lines = results_path.read_text(encoding="utf-8").splitlines()
    assert results_lines[177].strip() == "elif availability.message:"
    assert results_lines[178].strip() == "st.info(availability.message)"
    assert results_lines[179].strip() == "return availability"


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
    assert "Full-suite coverage for the delivered head remains unverified" in readme


def test_baseline_timeout_summary_preserves_reported_failure_and_unknowns() -> None:
    transcript = (EVIDENCE / "review-20261111/full-baseline-timeout.txt").read_text()
    assert any("F" in line and "[ 38%]" in line for line in transcript.splitlines())
    summary = (EVIDENCE / "review-20261111.md").read_text()
    assert "at least one test failure at 38%" in summary
    assert "failed test's identity and the timeout cause remain **unknown**" in summary
    assert "98% progress without a reported" not in summary
