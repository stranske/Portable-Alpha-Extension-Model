<!-- pr-preamble:start -->
> **Source:** Issue #957

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Issue #957 ensures sleeve constraint settings persist from the Scenario Wizard into ModelConfig, and can optionally be validated during a run.

#### Tasks
- [x] Add sleeve constraint fields to `ModelConfig`.
- [x] Add sleeve constraint defaults to `DefaultConfigView` and wire wizard state into config + YAML.
- [x] Validate sleeve constraints on run when enabled in the facade.
- [x] Add tests for wizard wiring and constraint validation.
- [x] Document sleeve constraint behavior for suggestor vs run validation.

#### Acceptance criteria
- [x] Wizard persists sleeve constraint values into YAML and validates into `ModelConfig`.
- [x] When `sleeve_validate_on_run=True`, violating runs fail deterministically.
- [x] Tests cover wiring + validation paths and pass in CI.
- [x] Documentation clarifies constraint behavior to avoid user confusion.

## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- [x] Verified sleeve wiring + validation tests via `pytest tests/test_wizard_config_wiring.py tests/test_sleeve_constraint_validation.py -m "not slow"`.

<!-- auto-status-summary:end -->

## Task Reconciliation
- [x] Reviewed recent commits for sleeve constraint wiring + validation updates.
- [x] Updated task checkboxes to reflect completed sleeve constraint work.
- [x] Synced PR body with Issue #957 scope and acceptance criteria.
