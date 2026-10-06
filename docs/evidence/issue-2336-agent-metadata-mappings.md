# Agent metadata mapping compatibility

The initial fix at `75d6ca0f` rejected empty non-dictionary `Mapping` objects
that the original falsy fallback accepted. `load_config` now checks `Mapping`
and converts accepted metadata to `dict`, preserving null/omitted defaults and
rejecting falsy non-mapping values. Four public-loader regressions exercise
empty and populated `UserDict` and `MappingProxyType` inputs, including the
zero-valued metadata field and normalized shares.

## Verified acceptance

- [x] The original 19 cases pass; the original adjacent selection passes 102
  cases. The expanded suite passes 23, and the expanded adjacent selection
  passes 106.
- [x] Original-source reproduction: 5 failed, 14 passed, excluding the four
  new compatibility cases. Those four cases all fail against `75d6ca0f`.
- [x] Five actual production mutations fail all 23 distinct nodes, with 24
  total failed executions. Source restored byte-identically; all 23 pass.
- [ ] The full-suite historical baseline/candidate counts and total coverage
  percentages have not been reproduced in this sandbox. Both attempts stalled
  at PNG export and were interrupted. No full-suite PASS is claimed.

The original adjacent selection is `tests/test_config.py`,
`tests/test_config_validation_paths.py`,
`tests/test_config_enhanced_validation.py`,
`tests/test_mean_conversion_config.py`, `tests/test_sweep_config.py`, and
`tests/test_config_agent_structure.py`, selecting
`-k 'not mapping_implementations'`. Remove that selector for the expanded run.
All test runs use `-m 'not slow and not live_llm'`.

## Mutation receipts

The accompanying JSON records exact failed node names, exit codes, source and
test hashes, and restoration. Mutations were made in a disposable copy of the
real production package, with its path supplied to a fresh pytest subprocess:

| Mutation | Selected behavior | Failed executions |
| --- | --- | ---: |
| `extra-guard` | Remove metadata shape validation | 7 |
| `entry-guard` | Replace the invalid-entry error with an empty entry | 3 |
| `missing-key-guard` | Substitute zero for missing required keys | 4 |
| `metadata-preservation` | Replace accepted metadata with a sentinel mapping | 9 |
| `model-entry` | Remove the `AgentConfig` dispatch branch | 1 |

The `AgentConfig` node is intentionally exercised by two mutations. The
historical reproduction confirms the original defect, while mutations confirm
the structural controls, including guards already correct in historical code.

## Formatting and scope

Both changed modules pass Black at line length 100. The complete 484-file
Black check passes using Black's normal `reformat_one` serially: the standard
concurrent invocation stalls under this sandbox's event-loop restrictions.
Black also required removal of one pre-existing extra blank line in
`tests/conftest.py`. Ruff and `git diff --check` pass.

Coverage configuration, exclusions, floors and skip markers remain unchanged.
The adjacent targeted command measures `pa_core.config` at 592/637 covered
statements (92.94%); that is a narrower test selection than the historical
full-suite measurement and does not establish the full-suite coverage target.
An expanded config/conversion/sweep/patch selection passes 183 tests and covers
593/637 statements (93.09%), still below the historical full-suite target.

The branch includes current main `7e7fbba8`, verified through GitHub's compare
API with zero commits behind. Local fetch/commit are unavailable because
`.git` is read-only. A source commit is preserved in an isolated writable Git
repository under `/tmp`; the tested source changes remain in this workspace.
GitHub publication was blocked because connector writes require approval and
this run's approval policy is `never`. The PR body/checklists cannot be updated
from this run; the verified checkboxes above provide the handoff record.
