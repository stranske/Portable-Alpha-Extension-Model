<!-- pr-preamble:start -->
> **Source:** Issue #1175

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Users can configure portfolios that violate constraints (negative weights, exceeding leverage limits, etc.) without clear feedback. Proactive validation would catch configuration errors early, suggest corrections for common mistakes, and improve user experience.

#### Tasks
- [x] Define common constraints for weight bounds
- [ ] Define common constraints for leverage
- [ ] Define common constraints for concentration
- [ ] Add validation hooks in portfolio construction
- [x] Generate suggestions for constraint violations
- [ ] Add `--validate-only` CLI flag
- [ ] Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages
- [ ] Define approach for: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages (verify: tests pass)
- [ ] Define scope for: Implement: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages (verify: tests pass)
- [ ] Implement focused slice for: Implement: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages (verify: tests pass)
- [ ] Validate focused slice for: Implement: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages (verify: tests pass)
- [ ] Define scope for: Validate: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages (verify: tests pass)
- [ ] Implement focused slice for: Validate: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages (verify: tests pass)
- [ ] Validate focused slice for: Validate: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages (verify: tests pass)
- [ ] Document supported constraints in `docs/constraints.md` with examples and usage guidelines
- [ ] Document supported constraints in `docs/constraints.md` with examples (verify: docs updated)
- [ ] Define scope for: Document supported constraints in `docs/constraints.md` with usage guidelines (verify: docs updated)
- [ ] Implement focused slice for: Document supported constraints in `docs/constraints.md` with usage guidelines (verify: docs updated)
- [ ] Validate focused slice for: Document supported constraints in `docs/constraints.md` with usage guidelines (verify: docs updated)

#### Acceptance criteria
- [ ] Invalid portfolios produce clear error messages
- [ ] Error messages suggest how to fix the violation
- [ ] Validation can run without executing full simulation
- [ ] Common constraints are pre-defined and documented
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- _None._

<!-- auto-status-summary:end -->
