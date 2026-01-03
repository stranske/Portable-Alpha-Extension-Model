<!-- pr-preamble:start -->
> **Source:** Issue #930

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Volatility regime selection already exists in concept, but its not first-class. Markets behave differently in calm vs stressed periods:
- Correlations spike during stress (diversification fails when needed most)
- Volatility clusters and mean-reverts
- Tail events are more likely during regime transitions

Making regime switching first-class enables realistic stress testing and scenario analysis.

#### Tasks
- [x] Define regime config structure: regimes with per-regime params
- [x] Implement regime-specific covariance matrices
- [x] Implement regime-specific correlation matrices
- [x] Add regime transition probability matrix
- [x] Integrate regime state into return generation
- [x] Add regime indicator to output (which regime was active each month)
- [x] Create preset regime configs (e.g., 2008 crisis, COVID shock)

#### Acceptance criteria
- [ ] Configs can specify multiple regimes with different vol/corr parameters
- [ ] Simulations switch regimes according to transition probabilities
- [x] Output includes regime state time series
- [x] Stressed regime produces higher correlations and volatility (configurable)
- [x] Unit tests verify regime switching logic and distributional properties
- [ ] ruff check and mypy pass

<!-- auto-status-summary:end -->
