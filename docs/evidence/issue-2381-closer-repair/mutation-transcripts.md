# Issue #2381 closer repair evidence

Check date: 2026-11-03 (America/Chicago).

Outcome: Gate-style full-suite coverage on ranked targets drops **209 → 20** missing
statements versus pre-#2384 round baseline (**189** reduction, ≥ ~180 goal).

Focused acceptance command:

```sh
.venv/bin/python -m pytest -q tests/test_coverage_round_2381.py tests/test_dashboard_results_previous_run.py
```

Stop rule: three failures at the same step or 90 minutes; one production mutation at a time
with byte restore before GREEN rerun.

See [gate-measurements.json](gate-measurements.json) and [ranking-provenance.json](ranking-provenance.json).

## Mutation samples

### `test_overlay_total_return_computes_from_summary_with_agent_total`

Mutation: append `+ 0.5` to `_overlay_total_return` return in `dashboard/pages/4_Results.py`.

RED: `FAILED ... AssertionError` (`0.58 != 0.08`).

GREEN after restore: `1 passed`.

### `test_results_main_renders_metrics_plots_and_download_for_available_workbook`

Mutation: change expected metric label `Tracking Error` → `Broken Error` in the test only
(detects loss of assertion wiring).

RED: `AssertionError` on metrics list equality.

GREEN after restore: `1 passed`.
