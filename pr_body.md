<!-- pr-preamble:start -->
> **Source:** Issue #957

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Issue #957 exposes advanced ModelConfig settings in the Streamlit Scenario Wizard and persists them into the wizard-generated YAML.

#### Tasks
- [x] Update `DefaultConfigView` to include advanced ModelConfig fields using the CLI value sets.
- [x] Add an advanced simulation settings UI section and wire values into the wizard config.
- [x] Emit advanced fields into `_build_yaml_from_config()`.
- [x] Cover advanced fields in `tests/test_wizard_schema.py` and `tests/test_wizard_config_wiring.py`.
- [x] Add regime switching wiring tests for wizard YAML output.
- [x] Document regime switching in the user guide.
- [x] Update wizard coverage documentation.

#### Acceptance criteria
- [x] Wizard UI exposes the advanced settings listed in Scope.
- [x] `_build_yaml_from_config()` includes these settings in its output dict.
- [x] The resulting YAML dict validates into `ModelConfig` with the expected values.
- [x] `tests/test_wizard_schema.py` and `tests/test_wizard_config_wiring.py` cover all added fields and pass in CI.
- [x] Documentation notes the new coverage.

## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- [x] Verified wizard schema/config wiring tests via `pytest tests/test_wizard_schema.py tests/test_wizard_config_wiring.py -m "not slow"`.
- [x] Verified wizard regime validation tests via `pytest tests/test_wizard_regime_wiring.py -m "not slow"`.

<!-- auto-status-summary:end -->

## Task Reconciliation
- [x] Reviewed recent commits for wizard advanced settings and correlation repair updates.
- [x] Updated task checkboxes to reflect completed wizard wiring work.
- [x] Synced PR body with Issue #957 scope and acceptance criteria.
- [x] Reconciled regime switching wiring/docs updates from recent commits.
