# Issue #2360 mutation transcripts

Current acceptance recheck: [acceptance.md](./acceptance.md), checked
**2026-11-07 UTC**. The original eight cases were reproduced at #2372's squash
merge with [exact mutation definitions](./mutation-plan.json),
[commands, exits and restoration hashes](./recheck/mutations/mutation-replay.json),
and full `01-red.txt` / `01-green.txt` through `08-red.txt` / `08-green.txt`
outputs in [recheck/mutations](./recheck/mutations/). Reproduce them with:

```sh
python scripts/replay_issue_2360_mutations.py --output /tmp/issue2360-mutation-replay
```

The following 2026-11-02 receipts remain historical; their later wizard hash
must not be confused with the original merge's restored hash.

Checked **2026-11-02, America/Chicago**. Each row is one temporary edit to
`dashboard/pages/3_Scenario_Wizard.py`, a focused red run, byte-for-byte
restoration, and a green rerun. Machine-readable mapping:
[mutations.json](./mutations.json).

Command template:

```text
.venv/bin/python -m pytest -q -o addopts= tests/test_wizard_render_coverage.py::<node>
```

## Eight original cases (#2372)

| Node | Red exit | Named failure (abbrev.) | Green |
| --- | ---: | --- | --- |
| `test_progress_bar_marks_completed_current_and_future_steps` | 1 | `('info', '📍 2. Capital Allocation')` missing from calls | 1 passed |
| `test_capital_step_reports_allocation_status_and_balanced_chart[50.0-error]` | 1 | no `error` call with `Capital allocation` | 1 passed |
| `test_capital_step_reports_allocation_status_and_balanced_chart[60.0-success]` | 1 | `StopIteration` fetching `plotly_chart` | 1 passed |
| `test_capital_step_handles_uploaded_and_persisted_margin_schedules[True]` | 1 | `Schedule validated ✓` not in calls | 1 passed |
| `test_capital_step_handles_uploaded_and_persisted_margin_schedules[False]` | 1 | no `info` with `Using schedule at` | 1 passed |
| `test_sleeve_suggestor_explains_missing_or_empty_results[None]` | 1 | unexpected `warning` in calls | 1 passed |
| `test_sleeve_suggestor_explains_missing_or_empty_results[empty]` | 1 | `warning` vs `info` for empty suggestions | 1 passed |
| `test_margin_schedule_loader_hides_invalid_schedule_errors` | 1 | `schedule is None` expected, got DataFrame | 1 passed |

## Nine later cases (#2411; excluded from #2372 attribution)

| Node | Red exit | Named failure (abbrev.) | Green |
| --- | ---: | --- | --- |
| `test_sleeve_suggestor_ranks_frontier_and_applies_confirmed_selection` | 1 | capital tuple mismatch after apply | 1 passed |
| `test_sleeve_suggestor_records_results_when_frontier_calculation_fails` | 1 | `sleeve_frontier` not empty | 1 passed |
| `test_sleeve_suggestor_falls_back_when_constraint_scope_is_unknown` | 1 | scope `'total'` vs `'sleeves'` | 1 passed |
| `test_sleeve_suggestor_stores_frontier_when_calculation_succeeds` | 1 | frontier not identical / chart missing | 1 passed |
| `test_correlation_step_warns_for_an_indefinite_matrix` | 1 | no `warning` with `positive definite` | 1 passed |
| `test_correlation_step_confirms_a_valid_matrix` | 1 | success text mismatch | 1 passed |
| `test_review_step_shows_financing_and_differences_before_a_disabled_run` | 1 | financing markdown heading mismatch | 1 passed |
| `test_review_step_shows_schedule_financing_and_matching_previous_config` | 1 | matches-last-run info text mismatch | 1 passed |
| `test_main_keeps_wizard_available_when_uploaded_yaml_is_invalid` | 1 | no `error` call after invalid YAML | 1 passed |

Restored wizard SHA-256: `5ba0c81b79b74f8128ac57346918fa7bbbc15b58e7fcf202d3e875ecea1cde98`.
