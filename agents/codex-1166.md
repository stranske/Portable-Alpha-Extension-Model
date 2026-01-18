<!-- bootstrap for codex on issue #1166 -->

## PR Tasks and Acceptance Criteria

**Progress:** 10/37 tasks complete, 27 remaining

### IMPORTANT: Task Reconciliation Required

The previous iteration changed 5 file(s) but did not update task checkboxes.

Before continuing, you MUST:
1. Review the recent commits to understand what was changed
2. Determine which task checkboxes should be marked complete
3. Update the PR body to check off completed tasks
4. Then continue with remaining tasks

Failure to update checkboxes means progress is not being tracked properly.

### Scope
Users can configure portfolios that violate constraints (negative weights, exceeding leverage limits, etc.) without clear feedback. Proactive validation would:
Catch configuration errors early
Suggest corrections for common mistakes
Improve user experience

Users can configure portfolios that violate constraints (negative weights, exceeding leverage limits, etc.) without clear feedback. Proactive validation would:
Catch configuration errors early
Suggest corrections for common mistakes
Improve user experience

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Define common constraints (weight bounds, leverage, concentration)
- [x] Define common constraints for weight bounds (verify: confirm completion in repo)
- [x] Define common constraints for leverage (verify: confirm completion in repo)
- [x] Define common constraints for concentration (verify: confirm completion in repo)
- [x] Add validation hooks in portfolio construction
- [ ] Generate suggestions for constraint violations
- [ ] Add `--validate-only` CLI flag
- [ ] Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages.
- [ ] Define approach for: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages. (verify: tests pass)
- [ ] Define scope for: Implement: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages. (verify: tests pass)
- [ ] Implement focused slice for: Implement: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages. (verify: tests pass)
- [ ] Validate focused slice for: Implement: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages. (verify: tests pass)
- [ ] Define scope for: Validate: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages. (verify: tests pass)
- [ ] Implement focused slice for: Validate: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages. (verify: tests pass)
- [ ] Validate focused slice for: Validate: Implement `ConstraintValidator` class to validate portfolio constraints with unit tests ensuring correct error messages. (verify: tests pass)
- [ ] Document supported constraints in `docs/constraints.md` with examples and usage guidelines.
- [ ] Document supported constraints in `docs/constraints.md` with examples (verify: docs updated)
- [ ] Define scope for: Document supported constraints in `docs/constraints.md` with usage guidelines. (verify: docs updated)
- [ ] Implement focused slice for: Document supported constraints in `docs/constraints.md` with usage guidelines. (verify: docs updated)
- [ ] Validate focused slice for: Document supported constraints in `docs/constraints.md` with usage guidelines. (verify: docs updated)
- [x] Define common constraints (weight bounds, leverage, concentration)
- [x] Define common constraints for weight bounds (verify: confirm completion in repo)
- [x] Define common constraints for leverage (verify: confirm completion in repo)
- [x] Define common constraints for concentration (verify: confirm completion in repo)
- [ ] Implement `ConstraintValidator` class
- [x] Add validation hooks in portfolio construction
- [ ] Generate suggestions for constraint violations
- [ ] Add `--validate-only` CLI flag
- [ ] Document supported constraints

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [ ] Invalid portfolios produce clear error messages
- [ ] Error messages suggest how to fix the violation
- [ ] Validation can run without executing full simulation
- [ ] Common constraints are pre-defined and documented
- [ ] Invalid portfolios produce clear error messages
- [ ] Error messages suggest how to fix the violation
- [ ] Validation can run without executing full simulation
- [ ] Common constraints are pre-defined and documented
