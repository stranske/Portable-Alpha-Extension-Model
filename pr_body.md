<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Align CLI warnings, outputs, and documentation with current deprecation behavior and delegation flow.

#### Tasks
- [x] Refactor the tests in `tests/test_cli_commands.py` to extract expected stdout/stderr outputs into reference constants or golden files, then compare test output against these references.
- [x] Update tests in `tests/test_deprecation_warnings.py` to include commentary on the usage of the Python warnings module and prepare for future changes in output mechanisms.
- [ ] Add inline comments in `pa_core/__main__.py` and `pa_core/cli.py` to clearly document how CLI argument parsing and delegation work.
- [ ] Review and update documentation in `docs/cli_usage_guide.md` and `docs/pa_core_facade.md` to reflect the current implementation details.
- [ ] Refactor output comparisons in `tests/test_main.py` and `tests/test_pa_core_main.py` to use stored expected output files or constants rather than inline hardcoded strings.

#### Acceptance criteria
- [x] Unit tests in `tests/test_cli_commands.py` must compare CLI command outputs (stdout, stderr) and exit codes against predefined constants or golden files.
- [x] Unit tests in `tests/test_deprecation_warnings.py` must verify that deprecation warnings are emitted using the Python warnings module and do not appear in stdout or stderr.
- [ ] Inline comments in `pa_core/__main__.py` and `pa_core/cli.py` must clearly document the CLI argument parsing and delegation process.
- [ ] Documentation in `docs/cli_usage_guide.md` and `docs/pa_core_facade.md` must include detailed explanations of the delegation process, deprecation warning behavior, and any changes to argument parsing or exit codes.
- [ ] Unit tests in `tests/test_main.py` and `tests/test_pa_core_main.py` must use stored expected output files or constants for stdout and stderr comparisons.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- _None._

<!-- auto-status-summary:end -->
