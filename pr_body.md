<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Follow-up to align CLI expectations, inline documentation, and reference artifacts after PR #1082.

#### Tasks
- [x] Refactor `tests/test_main.py` to replace all hard-coded expected CLI outputs with references to constants defined in `tests/expected_cli_outputs.py` or designated golden files.
- [x] Review and update `tests/expected_cli_outputs.py` to include constants for every expected CLI output used in `tests/test_main.py`.
- [x] Add inline comments in `pa_core/__main__.py` and `pa_core/cli.py` that reference external expected output artifacts and document the CLI argument parsing and delegation process.
- [x] Cross-check the updates in `docs/cli_usage_guide.md` and `docs/pa_core_facade.md` to ensure they align with the code changes and test refactoring.

#### Acceptance criteria
- [x] All assertions in `tests/test_main.py` use constants from `tests/expected_cli_outputs.py` or designated golden files for output comparison.
- [x] The file `tests/expected_cli_outputs.py` contains constants for all expected CLI outputs used in `tests/test_main.py`.
- [x] Inline comments in `pa_core/__main__.py` and `pa_core/cli.py` explicitly reference external expected output artifacts and document the CLI argument parsing and delegation process.
- [x] The documentation in `docs/cli_usage_guide.md` and `docs/pa_core_facade.md` aligns with the code changes and test refactoring.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- _None._

<!-- auto-status-summary:end -->
