"""Constants for expected CLI output comparisons in tests."""

# Expected stdout/stderr lines asserted in tests/test_main.py.
# Keep these names unique so each CLI output expectation maps to one constant.
# tests/test_main.py
MAIN_BACKEND_STDOUT = "[BACKEND] Using backend: numpy\n"
# tests/test_main.py
MAIN_BACKEND_STDERR = ""
# tests/test_cli_commands.py, tests/test_pa_core_main.py
EMPTY_STDERR = MAIN_BACKEND_STDERR
