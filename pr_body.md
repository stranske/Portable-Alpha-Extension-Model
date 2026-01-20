<!-- pr-preamble:start -->
> **Source:** Issue #1183

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
PR #1188 addressed issue #1183 but verification identified concerns (verdict: **CONCERNS**). This follow-up addresses the remaining gaps with improved task structure.

#### Tasks
- [x] [#1188](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1188)
- [x] [#1183](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1183)

#### Acceptance criteria
- [x] Update the mismatch detection function to explicitly check the agent type and apply the tolerance of 1e-6 only for ExternalPA, ActiveExt, and InternalBeta (using beta_share) and for InternalPA (using alpha_share). Ensure Base/unknown agents are excluded from triggering mismatch flags unless documented differently.
- [x] Add unit tests that assert the Excel workbook produced by 'pa run' contains a sheet named 'AgentSemantics' and that the sheet is populated correctly with the required columns and rows for all built-in agents.
- [x] Enhance docs/UserGuide.md to include detailed examples and use cases for the 'AgentSemantics' sheet. This should clearly document how the sheet is generated and how mismatch detection works for each agent type.
- [x] Modify the code to ensure that when a run path does not call build_agent_semantics explicitly, the Excel export routine still attaches a non-empty '_agent_semantics_df' (or its equivalent) as the 'AgentSemantics' sheet.
- [x] Ensure that for unknown/custom agent types, the 'notes' field is always populated with a message that indicates semantics depend on the specific implementation. Add corresponding unit tests to verify this behavior.
- [x] Review the logic handling the attachment of the '_agent_semantics_df' DataFrame into RunArtifacts.inputs to make sure it does not lead to serialization issues. If necessary, convert the DataFrame to a JSON- or YAML-compatible format or restrict its use to transient in-memory operations.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- _None._

<!-- auto-status-summary:end -->
