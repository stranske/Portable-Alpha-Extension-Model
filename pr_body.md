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
- [ ] Define multi-objective optimization problem structure
- [ ] Implement constraint-aware optimization (TE, CVaR, breach, shortfall bounds)
- [ ] Generate efficient frontier across return vs risk trade-off
- [x] Identify Pareto-optimal sleeve combinations
- [ ] Create frontier visualization with constraint boundaries
- [ ] Add interactive frontier exploration in dashboard
- [ ] Export frontier points to workbook

#### Acceptance criteria
- [ ] Optimizer respects all specified constraints
- [ ] Frontier shows return vs TE trade-off with CVaR as color dimension
- [ ] Infeasible regions (constraint violations) are clearly marked
- [ ] At least 20 frontier points generated for smooth curve
- [x] Dashboard allows clicking frontier points to see sleeve weights
- [ ] Unit tests verify constraint satisfaction for all frontier points
- [ ] ruff check and mypy pass

<!-- auto-status-summary:end -->
