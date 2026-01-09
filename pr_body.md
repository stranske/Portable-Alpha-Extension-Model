<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Isolate per-run RNG usage across simulations and regime switching, with deterministic tests for seeded runs.

#### Tasks
- [x] Audit `pa_core/facade.py`, `pa_core/sim/paths.py`, and `pa_core/sim/regimes.py` to identify and replace any direct usage of global RNG functions with `np.random.Generator` instances.
- [x] Modify functions in `pa_core/facade.py`, `pa_core/sim/paths.py`, and `pa_core/sim/regimes.py` to accept an `np.random.Generator` parameter or instantiate a local generator using a deterministic seed when necessary.
- [x] Update the simulation initialization module to create a new, seeded `np.random.Generator` at the start of each simulation run and pass it to all downstream functions.
- [x] Enhance tests in `tests/test_simulations.py` to include cases that perturb the global RNG state and verify that simulation outcomes remain consistent when supplied the same seed.

#### Acceptance criteria
- [x] All functions in `pa_core/facade.py`, `pa_core/sim/paths.py`, and `pa_core/sim/regimes.py` use an `np.random.Generator` instance for random number generation.
- [x] Functions in `pa_core/facade.py`, `pa_core/sim/paths.py`, and `pa_core/sim/regimes.py` accept an `np.random.Generator` parameter or instantiate one locally with a deterministic seed.
- [x] A new `np.random.Generator` is instantiated at the start of each simulation run in `simulation_initialization.py` using a provided seed.
- [x] Tests in `tests/test_simulations.py` confirm that simulations with the same seed produce identical results, different seeds produce different results, and changes to the global RNG state do not affect results.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- _None._

<!-- auto-status-summary:end -->
