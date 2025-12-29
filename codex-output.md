Added reporting coverage for risk attribution and run-diff logic, then updated the PR task checklist after a focused coverage run highlighted the lowest-covered reporting files. Coverage and tests were run against the new cases to confirm the changes.

Details
- Added a risk-attribution expectations test alongside the existing return-attribution coverage in `tests/test_attribution_reporting.py`.
- Added run-diff config/metric delta coverage and empty-summary handling in `tests/test_reporting_run_diff.py`.
- Updated task progress and completed checkboxes in `codex-prompt.md` after the reporting-module coverage run.

Coverage notes
- Focused coverage run on reporting modules shows lowest coverage in `pa_core/reporting/export_packet.py` and `pa_core/reporting/sweep_excel.py` (0%), with `pa_core/reporting/excel.py` also low (8%); `pa_core/reporting/console.py` at 33%.
- Full-suite coverage attempts timed out at 120s.

Tests
- `python -m pytest tests/test_attribution_reporting.py tests/test_reporting_run_diff.py --cov=pa_core/reporting --cov-report=term-missing --cov-report=term`

Next steps (pick one)
1) Add tests for dashboard modules (next unchecked task).
2) Extend coverage in reporting by targeting `pa_core/reporting/export_packet.py` and `pa_core/reporting/sweep_excel.py` before retrying a broader coverage run.