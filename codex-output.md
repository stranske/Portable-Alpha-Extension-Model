Implemented config validation-path coverage and captured a targeted coverage snapshot (full suite coverage runs timed out).

Changes
- Added validation-path tests for config failures in `tests/test_config_validation_paths.py` (financing model, backend, analysis mode, share weights, theta bounds, ShortfallProb presence).

Coverage snapshot (subset run)
- Lowest/zero coverage in this run: `pa_core/__main__.py`, `pa_core/portfolio/aggregator.py`, `pa_core/presets.py`, `pa_core/reporting/export_packet.py`, `pa_core/reporting/run_diff.py`, `pa_core/sensitivity.py`, `pa_core/sim/sensitivity.py`, `pa_core/wizard_schema.py`.
- Very low coverage: `pa_core/reporting/attribution.py` (~10%), `pa_core/data/importer.py` (~10%), `pa_core/cli.py` (~22%).

Notes
- Full coverage runs via `make test-cov` and broader `pytest --cov` runs timed out; the subset run completed quickly to identify the lowest-covered modules for next tasks.
- `codex-prompt.md` was already modified when I started; I did not touch it.

Tests
- `python -m pytest tests/test_config_validation_paths.py -v`
- `python -m pytest tests/test_config.py tests/test_config_enhanced_validation.py tests/test_reporting.py tests/test_cli_edge_cases.py tests/test_cli_exception_handling.py tests/test_validate_cli.py tests/test_pa_cli_validate.py --cov=pa_core --cov-report=term-missing --cov-report=term`

Next steps (pick one)
1) Add tests for `pa_core.reporting` (target `pa_core/reporting/attribution.py`, `pa_core/reporting/run_diff.py`).
2) Add tests for dashboard modules next, then retry full coverage with a longer timeout or a pruned test list.