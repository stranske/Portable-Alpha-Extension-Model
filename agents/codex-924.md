<!-- bootstrap for codex on issue #924 -->

## PR Tasks and Acceptance Criteria

**Progress:** 11/12 tasks complete, 1 remaining

### IMPORTANT: Task Reconciliation Required

The previous iteration changed 1 file(s) but did not update task checkboxes.

Before continuing, you MUST:
1. Review the recent commits to understand what was changed
2. Determine which task checkboxes should be marked complete
3. Update the PR body to check off completed tasks
4. Then continue with remaining tasks

Failure to update checkboxes means progress is not being tracked properly.

### Scope
Historically the codebase had overlapping mechanisms:
- prepare_mc_universe vs draw_joint_returns
- CLI building params inline
- Orchestrator doing something slightly different
- Sweeps doing something slightly different

Consolidation has begun via shared param builder and "canonical return path"
messaging, but if any path still builds its own params dict (or passes
slightly different ones), you get "same config, different results" bugs.

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Audit all entry points that build simulation params (CLI, orchestrator, sweeps, sensitivity)
- [x] Identify any param-building code that bypasses the canonical builder
- [x] Refactor all paths to call the single canonical build_params() function
- [x] Refactor all paths to call the single canonical draw_returns() function
- [x] Refactor all paths to call the single canonical draw_financing() function
- [x] Mark prepare_mc_universe as deprecated (or remove if fully migrated)
- [x] Add runtime assertion that params came from canonical builder (via marker field)

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] All entry points (CLI, orchestrator, sweep, sensitivity) use identical param building
- [x] No inline param construction exists outside the canonical builder
- [x] Tests verify that CLI and orchestrator produce identical outputs for same config
- [x] Deprecation warnings appear for any legacy function usage
- [ ] ruff check and mypy pass
