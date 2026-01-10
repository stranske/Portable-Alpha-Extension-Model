<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Align CLI warnings, outputs, and documentation with current deprecation behavior and delegation flow.

#### Tasks
- [x] Modify `tests/test_deprecation_warnings.py` to verify that deprecation warnings are captured using the warnings module and assert that these warnings do not appear in stdout or stderr.
- [x] Refactor `tests/test_cli_commands.py` and `tests/test_main.py` to replace inline hardcoded expected output strings with references to constants in `tests/expected_cli_outputs.py` or golden files.
- [x] Update `docs/cli_usage_guide.md` and `docs/pa_core_facade.md` to include detailed explanations of the CLI delegation process, deprecation warning behavior, argument parsing changes, and exit code definitions.
- [x] Enhance inline comments in `pa_core/__main__.py` and `pa_core/cli.py` to document the CLI argument parsing and delegation process, including references to external expected outputs.

#### Acceptance criteria
- [x] Deprecation warnings are captured using the Python warnings module and do not appear in stdout or stderr during test execution.
- [x] All assertions in `tests/test_cli_commands.py` and `tests/test_main.py` use constants from `tests/expected_cli_outputs.py` or golden files for output comparison.
- [x] Documentation in `docs/cli_usage_guide.md` and `docs/pa_core_facade.md` includes detailed explanations of CLI delegation, deprecation warnings, argument parsing changes, and exit code definitions.
- [x] Inline comments in `pa_core/__main__.py` and `pa_core/cli.py` clearly document the CLI argument parsing and delegation process, including references to external expected outputs.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- _None._

<!-- auto-status-summary:end -->
