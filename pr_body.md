<!-- pr-preamble:start -->
> **Source:** Issue #956

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Clarify that `InternalBeta` in simulation refers to margin-backed capital, while attribution uses a residual sleeve, and document naming rules to prevent term collisions.

#### Tasks
- [x] Document the distinction for simulation InternalBeta.
- [x] Document the distinction for attribution InternalBeta.
- [x] Consider renaming attribution InternalBeta to `ResidualBeta`.
- [x] Consider renaming attribution InternalBeta to `UnexplainedBeta`.
- [x] Add inline comments in both files explaining the semantic difference.
- [x] Update any user-facing documentation that references InternalBeta.

#### Acceptance criteria
- [x] Define specific criteria for what constitutes a term collision, such as 'terms must have distinct meanings in the context of the codebase'.
- [x] Documentation must include examples and clear definitions of both concepts.
- [x] Code comments in `registry.py` and `attribution.py` explain the distinction.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

<!-- auto-status-summary:end -->
