<!-- bootstrap for codex on issue #914 -->

## PR Tasks and Acceptance Criteria

**Progress:** 15/15 tasks complete, 0 remaining

### Scope
The new generic agents `[{name, capital, beta_share, alpha_share, extra}]` structure (#841) is a great extensibility improvement, but it creates new correctness requirements. Current potential failure cases:

- **No benchmark agent exists**: A lot of code assumes "Base" exists (TE relative to Base, default benchmark name, dashboard expectations)
- **Duplicate names**: Mixed-mode config compilation can produce two "Base" entries if user also adds one manually
- **Capital/share inconsistencies**: Multiple overlapping capital knobs exist:
  - `capital` per agent
  - `beta_share/alpha_share` per agent
  - Legacy capital knobs that get compiled in
  - `InternalBeta` sometimes auto-injected based on margin needs

This is fertile ground for accidental over-allocation or nonsense exposures.

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Add validation: exactly one benchmark agent exists (or explicitly allow multiple with benchmark choice)
- [x] Add validation: agent names are unique (no duplicates)
- [x] Add validation: `capital >= 0` for all agents
- [x] Add validation: `beta_share, alpha_share ∈ [0,1]`
- [x] Add validation: `beta_share + alpha_share <= 1` (or document what >1 means)
- [x] Add validation or documentation: if capital weights should be meaningful, enforce `sum(agent.capital)` constraint
- [x] Add clear error messages for each validation failure
- [x] Write tests for each validation rule

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] Config validation catches missing benchmark agent with clear error
- [x] Config validation catches duplicate agent names with clear error
- [x] Config validation catches negative capital with clear error
- [x] Config validation catches share values outside [0,1] with clear error
- [x] Config validation catches share sum >1 with clear error (or documents the behavior)
- [x] Tests cover all edge cases
- [x] Error messages are actionable (tell user what to fix)
