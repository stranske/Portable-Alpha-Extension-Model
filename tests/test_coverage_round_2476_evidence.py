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

    gap_tests = _test_names(ROOT / "tests" / "test_coverage_round_2476_remaining_gaps.py")
    assert {
        "test_export_converts_dataframe_agent_semantics",
        "test_export_converts_tuple_agent_semantics_to_list",
        "test_pa_registry_unknown_subcommand_is_a_no_op",
        "test_render_comparison_panel_without_message_shows_no_info",
    } <= gap_tests


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
