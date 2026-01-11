<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Understanding how results change with input parameters requires running many scenarios manually. Automated parameter sweeps would enable systematic sensitivity analysis, identification of critical parameters, and optimal parameter selection.

#### Tasks
- [x] Define sweep configuration schema (parameter ranges, grid/random sampling)
- [x] Implement `SweepRunner` class to execute parameter combinations
- [x] Add result aggregation (mean, std, percentiles across sweep)
- [x] Create summary report generator for sweeps
- [x] Add CLI support: `pa sweep --config sweep.yml`
- [x] Document sweep configuration format

#### Acceptance criteria
- [x] Sweep config specifies parameters, ranges, and sampling method
- [x] `SweepRunner` executes all combinations efficiently
- [x] Summary statistics are computed across sweep results
- [x] CLI enables sweep execution without code changes
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- _None._

<!-- auto-status-summary:end -->
