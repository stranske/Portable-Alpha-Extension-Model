Added a pytest coverage gate so CI runs fail under 85%, and captured a focused coverage snapshot to identify the lowest-covered modules. The change is in `pyproject.toml`, where `addopts` now enforces coverage across `pa_core` and `dashboard` with `--cov-fail-under=85`.

Coverage snapshot (from the targeted run) shows the lowest-covered modules are still at or near 0%:
- `dashboard/cli.py`, `dashboard/pages/*`, `dashboard/validation_ui.py`
- `pa_core/__main__.py`, `pa_core/pa.py`, `pa_core/data/convert.py`, `pa_core/portfolio/*`
- `pa_core/reporting/export_packet.py`, `pa_core/reporting/run_diff.py`
- `pa_core/validate.py`, `pa_core/sensitivity.py`, `pa_core/sim/sensitivity.py`, `pa_core/wizard_schema.py`

Tests run:
- `pytest --cov=pa_core --cov=dashboard --cov-report=term-missing tests/test_cli_edge_cases.py tests/test_config_validation_paths.py tests/test_reporting.py tests/test_dashboard.py`

Commit:
- `Enforce coverage threshold in pytest`

Notes:
- Full-suite coverage runs are timing out locally; the snapshot above is from a focused subset. If you want, I can try the full suite with a longer timeout or selective skips.

Next steps (pick one):
1) I can attempt a full coverage run with extended timeout and identify the true lowest-covered modules.
2) I can start adding tests in the worst 0% modules (dashboard pages or `pa_core` entrypoints) to push toward 85%.