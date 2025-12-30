Updated the module entrypoint to use the shared backend resolver and emit the backend banner at startup, matching the CLI behavior and backend helper. This keeps backend selection consistent across entrypoints without changing simulation flow. The change is in `pa_core/__main__.py`.

Tests: `pytest tests/test_backend_cli_integration.py::TestBackendCLIIntegration::test_backend_numpy_cli_flag` (timed out at 120s).

Notes: set local git identity to `Codex <codex@local>` to allow committing.

Next steps (pick one):
1. Rerun the CLI integration test with a longer timeout.
2. Run the full backend CLI integration test file if you want broader coverage.