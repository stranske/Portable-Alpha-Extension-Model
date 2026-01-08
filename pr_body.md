<!-- pr-preamble:start -->
> **Source:** Issue #959

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
Clarify InternalBeta vs UnexplainedBeta usage and ensure attribution naming and CI gate validation are handled consistently.

#### Tasks
- [x] Review and update inline comments in `pa_core/reporting/attribution.py` and `registry.py` to clearly explain the difference between InternalBeta and UnexplainedBeta, including direct references to the documentation.
- [x] Enhance the documentation in `docs/UserGuide.md` by adding specific, concrete examples for both InternalBeta and UnexplainedBeta. Ensure that each term is accompanied by a standalone definition that clearly demonstrates how it differs from the other term.
- [x] Audit the codebase to verify if the term renaming from InternalBeta to UnexplainedBeta has been fully or partially implemented. Update all instances to consistently reflect the new term if the renaming is to be completed.
- [ ] Wait for and validate the results from the CI gate workflow `pr-00-gate.yml`. Add a check or logging mechanism to confirm that queued CI jobs are completed before finalizing the merge.

#### Acceptance criteria
- [x] All inline comments in `pa_core/reporting/attribution.py` and `registry.py` must clearly explain the difference between InternalBeta and UnexplainedBeta, with direct references to the documentation.
- [x] The `docs/UserGuide.md` file must include at least one concrete example and a clear, standalone definition for each term (InternalBeta and UnexplainedBeta) that distinguishes their functionality and usage.
- [x] All instances of InternalBeta in the codebase must be updated to UnexplainedBeta if the renaming is to be completed, ensuring consistent terminology.
- [ ] The CI gate workflow `pr-00-gate.yml` must log completion of all queued CI jobs before allowing a merge.
## Related Issues
- [x] #1040
- [x] #958
## References
- [ ] _Not provided._

## Notes
- needs-human: Update `.github/workflows/pr-00-gate.yml` to log completion of queued CI jobs before merge; requires `agent-high-privilege`.

<!-- auto-status-summary:end -->
