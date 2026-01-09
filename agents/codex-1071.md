<!-- bootstrap for codex on issue #1071 -->

## PR Tasks and Acceptance Criteria

**Progress:** 4/4 tasks complete, 0 remaining

### IMPORTANT: Task Reconciliation Required

The previous iteration changed 1 file(s) but did not update task checkboxes.

Before continuing, you MUST:
1. Review the recent commits to understand what was changed
2. Determine which task checkboxes should be marked complete
3. Update the PR body to check off completed tasks
4. Then continue with remaining tasks

Failure to update checkboxes means progress is not being tracked properly.

### Scope
PR #1067 addressed issue #1065, but verification identified concerns despite a verdict of PASS.
This follow-up addresses the remaining gaps with an improved task structure to ensure robustness
and maintainability.

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Update `tests/test_cli_commands.py` to use regex or partial-string matching for non-critical output (e.g., help messages) instead of hardcoding full expected strings.
- [x] Enhance `tests/test_deprecation_warnings.py` to include comments and assertions that specifically capture warnings using the Python warnings module. Add a note in the test file stating that any deviation (e.g., warnings emitted via logging or print) violates the documented behavior.
- [x] Insert inline comments in `pa_core/__main__.py` and `pa_core/cli.py` to explain the CLI delegation logic, argument parsing decisions, and the relationship of these choices to the documented behavior.
- [x] Review and update the documentation files (`docs/cli_usage_guide.md` and `docs/pa_core_facade.md`) to clearly describe the new delegation process, changes to argument parsing, expected output behavior, and deprecation warning strategy.

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] Unit tests in `tests/test_cli_commands.py` must use regex or partial-string matching for non-critical output such as help messages.
- [x] Unit tests in `tests/test_cli_commands.py` must verify that each CLI command and subcommand produces the expected stdout, stderr, and exit codes as defined by the pre-refactor behavior.
- [x] Unit tests in `tests/test_deprecation_warnings.py` must capture deprecation warnings using Python's warnings module and ensure they do not appear in stdout or stderr.
- [x] Comments in `tests/test_deprecation_warnings.py` must document the expected mechanism for warning capture and flag any deviations.
- [x] Inline comments in `pa_core/__main__.py` and `pa_core/cli.py` must explain CLI delegation logic, argument parsing decisions, and their relationship to documented behavior.
- [x] Documentation in `docs/cli_usage_guide.md` and `docs/pa_core_facade.md` must include detailed explanations of the new CLI delegation process, argument parsing changes, exit codes, and deprecation warning behavior.
