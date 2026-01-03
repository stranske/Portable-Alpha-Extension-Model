<!-- bootstrap for codex on issue #938 -->

## PR Tasks and Acceptance Criteria

**Progress:** 13/13 tasks complete, 0 remaining

### Scope
Risk metric semantics are improving (explicit breach probability modes, horizon-adjusted thresholds, CVaR definitions), but ambiguity remains:

- `breach_count` is "count for path 0" - defensible for debug but misleading as a reported metric (users think its expected breaches across sims)
- CVaR is computed over flattened returns (months x paths) = "monthly CVaR across all draws", not "horizon CVaR of terminal outcomes"
- Both interpretations are useful but not interchangeable

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Audit all metrics in `pa_core/sim/` and `pa_core/portfolio/` for their semantic family
- [x] Rename or namespace metrics: `monthly_*` vs `terminal_*` prefix convention
- [x] Document each metrics semantic meaning in docstrings
- [x] Update output workbook column headers to reflect metric family
- [x] Fix `breach_count` labeling to clarify its "path 0 only" nature
- [x] Split CVaR into `cvar_monthly` (across all draws) and `cvar_terminal` (terminal outcomes only)
- [x] Add semantics test: "terminal CVaR gets worse with heavier tails under t-copula"

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] All metrics have explicit `monthly_` or `terminal_` prefix in output names
- [x] `breach_count` is either renamed to `breach_count_path0` or removed from default outputs
- [x] CVaR has two variants with different semantic meanings
- [x] Unit tests verify terminal CVaR increases with heavier tail distributions
- [x] Output workbook clearly distinguishes metric families
- [x] `ruff check` and `mypy` pass
