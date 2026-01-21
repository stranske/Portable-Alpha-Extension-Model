<!-- pr-preamble:start -->
> **Source:** Issue #1187

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
PR #1205 addressed issue #1187 but verification identified concerns (verdict: **CONCERNS**). This follow-up addresses the remaining gaps with improved task structure.

#### Tasks
- [x] [#1205](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1205)
- [x] [#1187](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1187)

#### Acceptance criteria
- [x] Refactor `simulate_regime_paths()` in `pa_core/sim/regimes.py` to remove any Python-level loops iterating over regimes and replace them with equivalent vectorized NumPy operations.
- [x] Update `simulate_regime_paths()` to use `np.random.Generator` with a fixed seed parameter for deterministic behavior.
- [x] Implement a deterministic snapshot test for `simulate_regime_paths()` in `tests/test_regime_switching.py` using a fixed seed and small dimensions.
- [x] Remove unrelated changes from the PR, such as scenario wizard UI, schema/config/templates, and additional test suites.

## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- Verified regime switching tests via `pytest tests/test_regime_switching.py -m "not slow"`.

<!-- auto-status-summary:end -->

## Task Reconciliation
- [x] Reviewed recent commits for regime switching changes.
- [x] Updated task checkboxes to reflect completed regime switching work.
- [x] Removed unrelated wizard/config/template changes and extra test updates.
