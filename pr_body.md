<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Align index/asset validation across static asset definitions, runtime schema checks, and calibration scenario generation so the index never appears in assets and correlations avoid duplicate pairs.

#### Tasks
- [x] Add unit tests in `tests/test_calibration_module.py` to verify that when assets include index.id, the Scenario validation function raises a ValueError with the message 'Assets must not include index.id'.
- [x] Modify `pa_core/schema.py` to raise a ValueError with the message 'Assets must not include index.id' when assets contain index.id.
- [x] Expand unit tests for `CalibrationResult.to_scenario()` in `tests/test_calibration_module.py` to cover edge cases including absence of index, single and multiple occurrences, and different orderings.
- [x] Update `docs/UserGuide.md` to explain the relationship between 'index' and 'assets', detailing the behavior of Scenario validation and `CalibrationResult.to_scenario()`, with examples.
- [ ] Review and update the CI workflow configuration in `pr-00-gate.yml` to ensure it triggers the complete test suite, including new tests. (needs-human)

#### Acceptance criteria
- [x] The Scenario validation function raises a ValueError with the exact message 'Assets must not include index.id' when 'index.id' is present in assets.
- [x] The `pa_core/schema.py` raises a ValueError with the message 'Assets must not include index.id' when assets contain index.id.
- [x] `CalibrationResult.to_scenario()` returns a scenario object whose assets do not include any entry for index in all test cases.
- [x] The `docs/UserGuide.md` includes a section explaining the relationship between 'index' and 'assets', detailing the behavior of Scenario validation and `CalibrationResult.to_scenario()`, with examples.
- [ ] The CI workflow configuration in `pr-00-gate.yml` triggers the complete test suite, including new tests. (needs-human)
#### Notes
- needs-human: Updating `.github/workflows/pr-00-gate.yml` is blocked by repo policy; please apply the workflow change in a high-privilege run.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

<!-- auto-status-summary:end -->
