<!-- pr-preamble:start -->
> **Source:** Issue #1195

<!-- pr-preamble:end -->

<!-- auto-status-summary:start -->
## Automated Status Summary
#### Scope
PR #1196 addressed issue #1195 but verification identified concerns (verdict: **CONCERNS**). This follow-up addresses the remaining gaps with improved task structure to ensure correct mismatch detection and data serialization.

#### Tasks
- [x] [#1196](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1196)
- [x] [#1195](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/1195)

#### Acceptance criteria
- [x] Add explicit conditional checks in `pa_core/reporting/agent_semantics.py` to ensure that mismatch detection is applied only to ExternalPA, ActiveExt, InternalBeta, and InternalPA agents, while Base/unknown agents always return a mismatch flag as false.
- [x] Update `pa_core/reporting/excel.py` and `pa_core/facade.py` to generate a non-empty agent semantics DataFrame and convert it to a JSON/YAML-compatible structure when `build_agent_semantics` is not called, ensuring it is attached as the 'AgentSemantics' sheet.
- [x] Enhance tests in `tests/test_reporting_agent_semantics.py` to verify that unknown/custom agent types have their 'notes' field set to 'Semantics depend on the specific agent implementation' and to simulate a run where `build_agent_semantics` is not called, asserting that the Excel export still includes a non-empty 'AgentSemantics' DataFrame.
- [x] Modify the code where `_agent_semantics_df` is attached to `RunArtifacts.inputs` to convert the raw pandas DataFrame into a serializable structure using methods such as `DataFrame.to_dict`.
## Related Issues
- [ ] _Not provided._
## References
- [ ] _Not provided._

## Notes
- Verified task checkboxes after latest test update.
- Added coverage for empty agent semantics dict export handling.
- Added facade export coverage for the AgentSemantics sheet.
- Added coverage for margin-driven InternalBeta semantics insertion.
- Confirmed task reconciliation after agent semantics serialization handling.
- Reconfirmed task checkboxes after series-based agent semantics serialization coverage.
- Reverified task reconciliation after list-of-series agent semantics serialization handling.
- Added tuple-of-series serialization coverage for agent semantics inputs.
- Added Excel export coverage for tuple-of-series AgentSemantics inputs.
- Added mismatch-flag guard coverage when total fund capital is zero.
- Added list-of-dataframes serialization coverage for agent semantics inputs.
- Added Excel export coverage for list-of-dataframes AgentSemantics inputs.
- Added tuple-of-dataframes serialization coverage for agent semantics inputs.
- Added Excel export coverage for tuple-of-dataframes AgentSemantics inputs.
- Reconfirmed task reconciliation after tuple-of-dicts serialization coverage.
- Added export coverage when total_fund_capital is missing from AgentSemantics inputs.
- Added zero-total-capital mismatch coverage for ActiveExt/custom agents.
- Added Excel export coverage for tuple-of-dicts AgentSemantics inputs.
- Reconciled task checkboxes after reviewing recent agent semantics export updates.
- Reverified task checkboxes after numpy scalar serialization handling updates.
- Verified AgentSemantics sheet content via facade export test.
- Normalized list-of-dicts AgentSemantics numpy scalars on export.
- Normalized nested list numpy scalars in AgentSemantics serialization.
- Normalized nested dict numpy scalars in AgentSemantics serialization.
- Normalized numpy array AgentSemantics values during serialization.
- Reverified task checkboxes after numpy array AgentSemantics handling.
- Added NaN total capital handling coverage for agent semantics mismatch flags.
- Reconfirmed task checkboxes after NaN agent semantics serialization handling.
- Reverified task checkboxes after invalid agent capital/share handling coverage.
- Added tuple-of-numpy-scalars serialization coverage for AgentSemantics inputs.
- Reconciled task checkboxes after reviewing non-finite margin requirement handling.

<!-- auto-status-summary:end -->
