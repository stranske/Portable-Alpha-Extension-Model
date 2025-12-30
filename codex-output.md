Centralized the parquet-missing hint for dashboard reads/writes and added tests to verify ImportError messaging and CSV sidecar creation for history/path loading in `dashboard/app.py`, `tests/test_dashboard_app_helpers.py`, and `tests/test_dashboard_run_history.py`.

Tests run:
- `pytest tests/test_dashboard_app_helpers.py tests/test_dashboard_run_history.py`

Next steps:
1. Run the full dashboard test suite if you want broader coverage: `pytest tests/test_dashboard_*.py`
2. Run the full test suite if required by CI.