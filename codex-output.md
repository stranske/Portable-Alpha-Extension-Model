Added focused CLI edge-case tests that exercise early-exit and error paths without running heavy simulations, covering invalid index inputs, sleeve suggestion short-circuiting, and JSON logging setup failure behavior in `tests/test_cli_edge_cases.py`. These new tests use lightweight monkeypatching to keep execution fast while hitting the specific branches in `pa_core.cli` that were previously untested.

**Coverage Notes**
- Ran a targeted coverage report; the full suite timed out, so this is based on a small subset.
- Lowest-covered modules from that report include `pa_core/cli.py` (~4%), multiple `pa_core/reporting/*` modules (0%), and most `dashboard/pages/*` (single-digit %). Command used: `pytest --cov=pa_core --cov=dashboard --cov-report=term-missing tests/test_cli_exception_handling.py tests/test_config.py tests/test_reporting.py tests/test_dashboard_pages.py`.

Tests run:
- `pytest tests/test_cli_edge_cases.py`

Next steps (pick one):
1. Add targeted tests for `pa_core.config` validation paths.
2. Add tests for `pa_core.reporting` (start with `export_packet.py`/`attribution.py`).
3. Add dashboard module tests to raise coverage on `dashboard/pages/*`.