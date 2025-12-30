Added explicit pyarrow-missing handling for parquet reads and writes in the dashboard, with a consistent `st.info` hint and ensured CSV fallback keeps working. Updated the parquet sidecar loader to surface the hint only on ImportError while still falling back to CSV/Excel when possible in `dashboard/app.py`. Updated the history save/load paths to share the same guidance for missing parquet support in `dashboard/app.py`.

Tests:
- `pytest tests/test_dashboard_run_history.py`

Notes:
- Unrelated modified file remains: `codex-prompt.md`.

Next steps:
1) Run a wider test pass if you want broader coverage (e.g., `pytest`).
2) Smoke-test the dashboard to see the `st.info` hint when pyarrow is absent.