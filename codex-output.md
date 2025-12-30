I couldn’t find a failing check to fix. The run metadata says “Failing jobs: none reported,” and `autofix_report_enriched.json` only lists `tests/test_dashboard_app_helpers.py` without error details. I didn’t make any code changes.

If you can share the specific CI log snippet (or a failing job name), I can target the fix. Otherwise, a good next step is to rerun the CI job or run the relevant local test suite to surface the actual failure.

Suggestions:
1) Paste the failing CI log excerpt (or job name) so I can pinpoint the root cause.
2) Re-run the CI job to see if a concrete failure appears.