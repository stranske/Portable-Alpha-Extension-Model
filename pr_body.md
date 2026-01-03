<!-- pr-preamble:start -->
> **Source:** Issue #947

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Stress testing is ad-hoc - users manually adjust parameters to simulate stress scenarios. Need:
- Consistent stress presets that are reusable and documented
- Clear reporting of what broke under stress (which constraint failed, which sleeve drove breaches)
- Ability to compare baseline vs stressed outcomes

#### Tasks
- [x] Create stress preset library with named scenarios
- [x] Implement stress application function that modifies base config
- [x] Add constraint failure detection and reporting
- [x] Identify which sleeve drove each breach
- [ ] Create stress comparison output (baseline vs stressed side-by-side)
- [ ] Add stress summary dashboard widget
- [ ] Document preset scenarios in user guide

#### Acceptance criteria
- [x] At least 5 preset stress scenarios available (e.g., rate shock, equity crash, correlation spike)
- [ ] Stress output clearly shows which constraints failed
- [ ] Per-sleeve breach attribution is included
- [ ] Comparison table shows baseline vs stressed metrics
- [x] Unit tests verify stress presets produce expected metric changes
- [ ] ruff check and mypy pass

<!-- auto-status-summary:end -->
