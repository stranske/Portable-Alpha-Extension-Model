Added targeted tests for CLI edge cases, config validation, and reporting exports to cover more branches in `pa_core` and `pa_core.reporting`, plus a focused coverage run to identify gaps. Updates are in `tests/test_cli_edge_cases.py`, `tests/test_config_validation_paths.py`, and `tests/test_reporting_exports.py`.

Coverage report highlights (lowest coverage from the run):
- `dashboard/*` modules are at 0% across app and pages.
- `pa_core/pa.py`, `pa_core/validate.py`, `pa_core/sensitivity.py`, `pa_core/sim/sensitivity.py`, `pa_core/portfolio/*`, `pa_core/presets.py`, `pa_core/wizard_schema.py` are 0%.
- `pa_core/cli.py` remains low (~24%), and several `pa_core/reporting` modules still lag (e.g., `run_diff.py`, `console.py`).

Tests run:
- `pytest tests/test_cli_edge_cases.py tests/test_config_validation_paths.py tests/test_reporting_exports.py --cov=pa_core --cov-report=term-missing`

Note: `codex-prompt.md` is modified in the working tree but wasn’t touched by me. Do you want me to leave it as-is or investigate?

Next steps (pick one):
1) Add tests for the lowest-covered core modules (e.g., `pa_core/pa.py`, `pa_core/validate.py`, `pa_core/cli.py`) and re-run coverage.  
2) Update CI to enforce the 85% threshold after expanding coverage enough to meet the target.