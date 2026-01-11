<!-- bootstrap for codex on issue #964 -->

## PR Tasks and Acceptance Criteria

**Progress:** 5/11 tasks complete, 6 remaining

### IMPORTANT: Task Reconciliation Required

The previous iteration changed **3 file(s)** but did not update task checkboxes.

**Before continuing, you MUST:**
1. Review the recent commits to understand what was changed
2. Determine which task checkboxes should be marked complete
3. Update the PR body to check off completed tasks
4. Then continue with remaining tasks

_Failure to update checkboxes means progress is not being tracked properly._

### Scope
Currently, comparing multiple scenarios requires manually inspecting output files or metrics. An interactive visualization would enable:
- Quick comparison of scenario outcomes
- Sensitivity analysis exploration
- Better communication of results to stakeholders

### Tasks
Complete these in order. Mark checkbox done ONLY after implementation is verified:

- [x] Create `pa_core/viz/` module structure
- [x] Implement scenario comparison plots (matplotlib/plotly)
- [x] Add return distribution comparison (histogram, density)
- [ ] Add risk metric bar charts (VaR, CVaR, max drawdown)
- [ ] Add regime timeline visualization
- [ ] Integrate with existing `export()` workflow
- [ ] Add CLI flag for generating visualizations

### Acceptance Criteria
The PR is complete when ALL of these are satisfied:

- [x] `pa_core.viz.compare_scenarios(results_list)` produces comparison plots
- [ ] Plots can be saved to file or displayed interactively
- [ ] Documentation includes visualization examples
- [x] Unit tests verify plot generation without display
