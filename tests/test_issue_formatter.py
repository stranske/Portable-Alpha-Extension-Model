from scripts.langchain import issue_formatter


def _section_lines(body: str, header: str) -> list[str]:
    lines = body.splitlines()
    try:
        header_idx = next(i for i, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        return []
    end_idx = next(
        (
            i
            for i in range(header_idx + 1, len(lines))
            if lines[i].startswith("## ") and lines[i].strip() != header
        ),
        len(lines),
    )
    return [line.strip() for line in lines[header_idx + 1 : end_idx] if line.strip()]


def test_format_issue_body_uses_checked_placeholder_for_missing_tasks() -> None:
    result = issue_formatter.format_issue_body("", use_llm=False)
    formatted = result["formatted_body"]

    tasks_lines = _section_lines(formatted, "## Tasks")
    acceptance_lines = _section_lines(formatted, "## Acceptance Criteria")

    assert "- [x] _Not provided._" in tasks_lines
    assert "- [x] _Not provided._" in acceptance_lines
    assert "- [ ] _Not provided._" not in tasks_lines
    assert "- [ ] _Not provided._" not in acceptance_lines
