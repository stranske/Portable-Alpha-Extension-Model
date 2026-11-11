"""Guard the 41 production-mutation records for issue #2476.

Each record in docs/evidence/issue-2476/mutation-transcripts.md must name a
real test, show a red run then a green run, and prove the production file was
restored (sha256 after restoration equals sha256 before mutation). A mutation
whose removed text is not unique in its production file, or which only adds
lines, must carry a ``Site:`` line identifying the exact mutation site.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "docs" / "evidence" / "issue-2476" / "mutation-transcripts.md"
TEST_FILES = [
    ROOT / "tests" / "test_coverage_round_2381.py",
    ROOT / "tests" / "test_coverage_round_2476_facade.py",
]


def _measured_source(path: str) -> str:
    snapshot = json.loads(
        (TRANSCRIPTS.parent / "mutation-source-snapshot.json").read_text(encoding="utf-8")
    )
    revision = re.search(r"code revision `([0-9a-f]{40})`", TRANSCRIPTS.read_text())
    assert revision and snapshot["revision"] == revision.group(1)
    return snapshot["sources"][path]


def _records() -> list[tuple[str, str]]:
    text = TRANSCRIPTS.read_text(encoding="utf-8")
    parts = re.split(r"\n## `", text)
    records = []
    for part in parts[1:]:
        heading, _, body = part.partition("`")
        if "- Red exit:" in body:
            records.append((heading, body))
    return records


def _test_names() -> set[str]:
    names: set[str] = set()
    for path in TEST_FILES:
        names.update(re.findall(r"^def (test_\w+)\(", path.read_text(encoding="utf-8"), re.M))
    return names


def _base_test_name(heading: str) -> str:
    name = heading.split("[", 1)[0].split(" (", 1)[0].strip()
    return name


def test_mutation_transcripts_have_41_complete_records() -> None:
    records = _records()
    assert len(records) == 41
    known_tests = _test_names()
    for heading, body in records:
        assert _base_test_name(heading) in known_tests, heading
        before = re.search(r"sha256 before: `([0-9a-f]{64})`", body)
        mutated = re.search(r"sha256 mutated: `([0-9a-f]{64})`", body)
        restored = re.search(r"sha256 restored: `([0-9a-f]{64})` \(matches\)", body)
        assert before and mutated and restored, heading
        assert before.group(1) == restored.group(1), heading
        assert mutated.group(1) != before.group(1), heading
        assert "- Red exit: 1" in body, heading
        assert "- Green exit: 0" in body, heading
        file_match = re.search(r"- File: `([^`]+)`", body)
        assert file_match, heading
        source = _measured_source(file_match.group(1))
        assert hashlib.sha256(source.encode("utf-8")).hexdigest() == before.group(1), heading
        assert _base_test_name(heading) in body.split("Command:", 1)[-1], heading


def _diff_lines(body: str) -> tuple[list[str], list[str]]:
    diff_match = re.search(r"```diff\n(.*?)```", body, re.S)
    assert diff_match
    removed = [line[1:] for line in diff_match.group(1).splitlines() if line.startswith("-")]
    added = [line[1:] for line in diff_match.group(1).splitlines() if line.startswith("+")]
    return removed, added


def test_mutation_transcripts_removed_lines_exist_in_production_file() -> None:
    for heading, body in _records():
        file_match = re.search(r"- File: `([^`]+)`", body)
        assert file_match, heading
        production_lines = {
            line.strip() for line in _measured_source(file_match.group(1)).splitlines()
        }
        removed, _added = _diff_lines(body)
        for line in removed:
            assert line.strip() in production_lines, (heading, line)


def test_mutation_transcripts_site_references_point_at_changed_lines() -> None:
    for heading, body in _records():
        site_match = re.search(r"- Site: `([^`:]+):(\d+)`", body)
        if site_match is None:
            continue
        file_match = re.search(r"- File: `([^`]+)`", body)
        assert file_match, heading
        assert site_match.group(1) == file_match.group(1), heading
        production = _measured_source(file_match.group(1)).splitlines()
        line_number = int(site_match.group(2))
        assert 1 <= line_number <= len(production), heading
        removed, added = _diff_lines(body)
        if not removed:
            continue
        changed = {line.strip() for line in removed + added}
        assert production[line_number - 1].strip() in changed, heading


def test_mutation_transcripts_disambiguate_non_unique_mutations() -> None:
    for heading, body in _records():
        file_match = re.search(r"- File: `([^`]+)`", body)
        assert file_match, heading
        raw_lines = _measured_source(file_match.group(1)).splitlines()
        stripped_lines = [line.strip() for line in raw_lines]
        diff_match = re.search(r"```diff\n(.*?)```", body, re.S)
        assert diff_match, heading
        removed = [line[1:] for line in diff_match.group(1).splitlines() if line.startswith("-")]
        ambiguous = not removed or any(
            raw_lines.count(line) != 1 and stripped_lines.count(line.strip()) != 1
            for line in removed
        )
        if ambiguous:
            assert "- Site: `" in body, heading


def test_historical_source_checks_ignore_current_checkout(monkeypatch) -> None:
    original_read = Path.read_text

    def reject_current_source(path, *args, **kwargs):
        if path.relative_to(ROOT).as_posix() in {
            "pa_core/facade.py",
            "pa_core/pa.py",
            "pa_core/llm/result_explain.py",
            "dashboard/pages/4_Results.py",
        }:
            raise AssertionError("historical evidence must not read current source")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_current_source)
    test_mutation_transcripts_have_41_complete_records()
    test_mutation_transcripts_removed_lines_exist_in_production_file()
    test_mutation_transcripts_site_references_point_at_changed_lines()
    test_mutation_transcripts_disambiguate_non_unique_mutations()
