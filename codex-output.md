Added a new test to confirm `save_history` writes `Outputs.csv` even when parquet writes succeed, and updated the task checkboxes now that the behavior is verified. This keeps the dashboard history export robust without requiring pyarrow while documenting completion in `codex-prompt.md`.

- Test coverage: `tests/test_dashboard_run_history.py` now stubs parquet writes and asserts both `Outputs.parquet` and `Outputs.csv` are created.
- Task tracking: checked off scope, tasks, and acceptance criteria in `codex-prompt.md`.

Tests run:
- `pytest tests/test_dashboard_run_history.py`

Next steps:
1. Run the broader dashboard tests if you want extra coverage: `pytest tests/test_dashboard_*.py`