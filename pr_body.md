<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Isolate per-run RNG usage across simulations and regime switching, with deterministic tests for seeded runs.

#### Tasks
- [x] Modify `pa_core/sim/paths.py` to replace module-level RNG usage with a local `np.random.Generator` instance. Update functions to accept a generator parameter if necessary.
- [x] Refactor `pa_core/sim/regimes.py` to eliminate module-level RNG usage by using a per-run `np.random.Generator` instance. Pass the generator as a parameter or instantiate one within the simulation run.
- [x] Update `pa_core/facade.py` to instantiate a new `np.random.Generator` at the start of each simulation run and ensure it is passed to all downstream functions.
- [x] Review and update existing tests in `tests/test_simulations.py` to explicitly validate RNG isolation by checking that simulations with the same seed produce identical results and those with different seeds produce different results.

#### Acceptance criteria
- [x] All functions in `pa_core/sim/paths.py` that previously used module-level RNG now accept an `np.random.Generator` instance as a parameter or create one locally.
- [x] All functions in `pa_core/sim/regimes.py` that previously used module-level RNG now accept an `np.random.Generator` instance as a parameter or create one locally.
- [x] A new `np.random.Generator` is instantiated at the start of each simulation run in `pa_core/facade.py` and passed to all downstream functions.
- [x] Tests in `tests/test_simulations.py` validate that simulations with the same seed produce identical results and those with different seeds produce different results.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- _None._

<!-- auto-status-summary:end -->
