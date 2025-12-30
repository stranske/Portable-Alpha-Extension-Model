Focused on dashboard coverage gaps by adding targeted unit tests for validation UI, CLI launch, and app helper fallbacks, and re-ran the dashboard coverage report to pinpoint the lowest-covered pages. Changes are in `tests/test_dashboard_validation_ui.py`, `tests/test_dashboard_cli.py`, and `tests/test_dashboard_app_helpers.py`.

Details on the coverage scan: the dashboard-only report still shows the lowest coverage in the page modules: `dashboard/pages/7_Run_Logs.py` at 0%, `dashboard/pages/3_Scenario_Wizard.py` at 6%, `dashboard/pages/5_Scenario_Grid.py` at 8%, `dashboard/pages/4_Results.py` at 11%, `dashboard/pages/2_Portfolio_Builder.py` at 14%, and `dashboard/pages/6_Stress_Lab.py` at 16%.

Tests run:
- `pytest tests/test_dashboard_validation_ui.py tests/test_dashboard_cli.py tests/test_dashboard_app_helpers.py --cov=dashboard --cov-report=term-missing`
- `pytest tests/test_dashboard* tests/test_dashboard_validation_ui.py tests/test_dashboard_cli.py tests/test_dashboard_app_helpers.py --cov=dashboard --cov-report=term-missing`

Note: full-suite coverage (`pytest --cov=pa_core --cov=dashboard --cov-report=term-missing -k "not golden"`) timed out at 240s; I used the dashboard-focused coverage report to satisfy the “identify lowest-covered modules” step.

Next steps:
1. Add tests for the lowest-coverage dashboard pages (`dashboard/pages/7_Run_Logs.py` and `dashboard/pages/3_Scenario_Wizard.py` are the biggest gaps).
2. Re-run the full coverage suite with a longer timeout to confirm the overall baseline once those tests land.