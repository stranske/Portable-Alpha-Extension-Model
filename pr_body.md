<!-- pr-preamble:start -->
> **Source:** Issue #932

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
The sleeve suggester currently optimizes for a single objective. Real portfolio construction requires multi-objective optimization:
- Maximize expected return
- Subject to: TE constraint, CVaR constraint, breach probability constraint, shortfall constraint
- Visualize the efficient frontier across these dimensions

#### Tasks
- [x] Define multi-objective optimization problem structure
- [x] Implement constraint-aware optimization (TE, CVaR, breach, shortfall bounds)
- [x] Generate efficient frontier across return vs risk trade-off
- [x] Identify Pareto-optimal sleeve combinations
- [x] Create frontier visualization with constraint boundaries
- [x] Add interactive frontier exploration in dashboard
- [x] Export frontier points to workbook

#### Acceptance criteria
- [x] Optimizer respects all specified constraints
- [x] Frontier shows return vs TE trade-off with CVaR as color dimension
- [x] Infeasible regions (constraint violations) are clearly marked
- [x] At least 20 frontier points generated for smooth curve
- [x] Dashboard allows clicking frontier points to see sleeve weights
- [x] Unit tests verify constraint satisfaction for all frontier points
- [x] ruff check and mypy pass

<!-- auto-status-summary:end -->
