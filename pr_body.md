<!-- pr-preamble:start -->
> **Source:** Issue #948

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Portfolio-level metrics are useful but dont answer "which sleeve is driving my results?" Users need:
- **Contribution to return**: Which sleeves add/subtract from total return
- **Contribution to tracking error**: Which sleeves drive deviation from benchmark
- **Contribution to tail risk**: Marginal CVaR contribution per sleeve

This enables informed sleeve allocation decisions and risk budgeting.

#### Tasks
- [ ] Implement return attribution: sleeve contribution to total portfolio return
- [ ] Implement TE attribution: sleeve contribution to tracking error (using covariance decomposition)
- [ ] Implement marginal CVaR: how much each sleeve contributes to portfolio tail risk
- [ ] Create attribution output table in workbook with per-sleeve breakdown
- [ ] Add visualization for attribution pie/bar charts
- [ ] Document attribution methodology in user guide

#### Acceptance criteria
- [ ] Return contributions sum to total portfolio return (within floating point tolerance)
- [ ] TE contributions are computed using proper covariance-based decomposition
- [ ] Marginal CVaR contributions sum to total CVaR (Euler decomposition property)
- [ ] Output workbook includes `sleeve_attribution` sheet
- [ ] Unit tests verify attribution math with known analytical cases
- [ ] `ruff check` and `mypy` pass

<!-- auto-status-summary:end -->
