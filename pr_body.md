<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Align index/asset validation across static asset definitions, runtime schema checks, and calibration scenario generation so the index never appears in assets and correlations avoid duplicate pairs.

#### Tasks
- [x] Define validation logic for rejecting index ID in assets.
- [x] Implement the validation in the `Scenario` model.
- [x] Write unit tests for the new validation.
- [x] Modify `CalibrationResult.to_scenario()` to filter out the index.
- [x] Define scope for: Write unit tests to ensure the index is not included in the assets list.
- [x] Implement focused slice for: Write unit tests to ensure the index is not included in the assets list.
- [x] Validate focused slice for: Write unit tests to ensure the index is not included in the assets list.
- [x] Document the expected relationship between index and assets.

#### Acceptance criteria
- [x] `Scenario` raises `ValidationError` with message `assets must not include index id 'IDX'` when `index.id` appears in assets.
- [x] `CalibrationResult.to_scenario()` never emits assets containing the index (e.g., series `IDX/A` yields assets `["A"]`).
- [x] Correlation validation rejects missing, duplicate, or extra pairs so each unique id combination appears exactly once.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

<!-- auto-status-summary:end -->
