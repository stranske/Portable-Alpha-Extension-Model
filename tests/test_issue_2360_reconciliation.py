"""Prepare current acceptance checkboxes only when the retained receipts verify."""

import pytest

from scripts import verify_issue_2360_evidence as verifier


@pytest.fixture
def body() -> str:
    tasks = "\n".join(f"- [ ] {item}" for item in verifier.CHECKLIST_ITEMS)
    return (
        "### Keepalive: OFF\n#### Tasks\n"
        + tasks
        + "\n<details>\n## Tasks\n"
        + tasks
        + "\n## Implementation Notes\nHistorical observations\n"
        "<details><summary>Original Issue</summary>\n"
        "## Tasks\n- [ ] Preserve this historical task\n</details>\n</details>\n"
    )


@pytest.fixture
def results() -> dict[str, tuple[str, str]]:
    return {name: ("PASS", "verified receipt") for name in verifier.CHECKS}


def test_reconciliation_preserves_history_and_is_idempotent(body, results):
    reconciled = verifier.reconcile_pr_body(body, results)
    current, _, historical = reconciled.partition("\n## Implementation Notes\n")
    assert current.count("- [x] ") == 22
    assert "- [ ] " not in current
    assert historical == body.partition("\n## Implementation Notes\n")[2]
    assert reconciled.startswith("### Keepalive: OFF\n")
    assert verifier.reconcile_pr_body(reconciled, results) == reconciled


@pytest.mark.parametrize("requirement", verifier.CHECKS)
@pytest.mark.parametrize("status", ["FAIL", "UNKNOWN"])
def test_reconciliation_rejects_any_unresolved_requirement(body, results, requirement, status):
    results[requirement] = (status, "missing proof")
    with pytest.raises(ValueError, match="all acceptance requirements"):
        verifier.reconcile_pr_body(body, results)


def test_reconciliation_rejects_incomplete_result_set(body):
    with pytest.raises(ValueError, match="all acceptance requirements"):
        verifier.reconcile_pr_body(body, {})


@pytest.mark.parametrize("breakage", ["unknown", "missing", "duplicate", "boundary"])
def test_reconciliation_rejects_unrecognized_checklist(body, results, breakage):
    line = f"- [ ] {verifier.CHECKLIST_ITEMS[0]}\n"
    if breakage == "unknown":
        body = body.replace(line, "- [ ] Unrelated task\n", 1)
    elif breakage == "missing":
        body = body.replace(line, "", 1)
    elif breakage == "duplicate":
        body = body.replace(line, line * 2, 1)
    else:
        body = body.replace("## Implementation Notes", "## Other Notes")
    with pytest.raises(ValueError, match="unrecognized current"):
        verifier.reconcile_pr_body(body, results)


def test_cli_prepares_reviewable_body_without_changing_input(tmp_path, body, capsys):
    source = tmp_path / "input.md"
    output = tmp_path / "output.md"
    source.write_text(body)
    assert verifier.main(["--reconcile-pr-body", str(source), "--output", str(output)]) == 0
    assert source.read_text() == body
    assert output.read_text().count("- [x] ") == 22
    assert "publication pending" in capsys.readouterr().out


def test_cli_does_not_overwrite_proposal_when_evidence_fails(tmp_path, body, monkeypatch, capsys):
    source = tmp_path / "input.md"
    output = tmp_path / "output.md"
    source.write_text(body)
    output.write_text("Previous reviewable proposal")
    monkeypatch.setattr(verifier, "verify", lambda _: {"CI": ("FAIL", "missing receipt")})
    assert verifier.main(["--reconcile-pr-body", str(source), "--output", str(output)]) == 1
    assert output.read_text() == "Previous reviewable proposal"
    assert "OVERALL FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("breakage", ["missing_input", "unknown_checklist", "output_directory"])
def test_cli_reports_preparation_errors_without_success(tmp_path, body, breakage, capsys):
    source = tmp_path / "input.md"
    output = tmp_path / "output.md"
    if breakage != "missing_input":
        source.write_text("Unrecognized checklist" if breakage == "unknown_checklist" else body)
    if breakage == "output_directory":
        output.mkdir()
    assert verifier.main(["--reconcile-pr-body", str(source), "--output", str(output)]) == 1
    report = capsys.readouterr().out
    assert "FAIL: checklist reconciliation:" in report
    assert "OVERALL FAIL" in report
    assert "OVERALL PASS" not in report
    assert not output.is_file()


@pytest.mark.parametrize("option", ["--reconcile-pr-body", "--output"])
def test_cli_requires_both_paths(tmp_path, option):
    with pytest.raises(SystemExit) as error:
        verifier.main([option, str(tmp_path / "body.md")])
    assert error.value.code == 2


def test_cli_refuses_to_overwrite_input(tmp_path, body):
    source = tmp_path / "body.md"
    source.write_text(body)
    with pytest.raises(SystemExit) as error:
        verifier.main(["--reconcile-pr-body", str(source), "--output", str(source)])
    assert error.value.code == 2
    assert source.read_text() == body
