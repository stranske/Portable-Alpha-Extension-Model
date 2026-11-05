# Issue #2413 closer repair — mutation transcripts
Interpreter: `/Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python`
Module: `tests/test_coverage_round_2413.py`

## 1. `test_main_routes_each_step_to_its_renderer[1]`
**Mutation:** Route step 1 through step 99 guard

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[1]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:134: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:27.350 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[1]
1 failed in 1.95s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.82s
```

## 2. `test_main_routes_each_step_to_its_renderer[2]`
**Mutation:** Route step 2 through step 99 guard

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[2]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:134: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:29.721 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[2]
1 failed in 0.91s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.88s
```

## 3. `test_main_routes_each_step_to_its_renderer[3]`
**Mutation:** Route step 3 through step 99 guard

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[3]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:134: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:32.165 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[3]
1 failed in 0.92s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.83s
```

## 4. `test_main_routes_each_step_to_its_renderer[4]`
**Mutation:** Route step 4 through step 99 guard

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[4]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:134: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:34.518 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[4]
1 failed in 0.90s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.82s
```

## 5. `test_main_routes_each_step_to_its_renderer[5]`
**Mutation:** Route step 5 through step 99 guard

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[5]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:134: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:36.878 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_routes_each_step_to_its_renderer[5]
1 failed in 0.90s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.82s
```

## 6. `test_main_cold_start_initializes_and_routes_to_step_1`
**Mutation:** Cold start lands on step 99 instead of 1

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_cold_start_initializes_and_routes_to_step_1
```

**RED tail:**
```
tests/test_coverage_round_2413.py:147: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:39.218 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_cold_start_initializes_and_routes_to_step_1
1 failed in 0.91s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.83s
```

## 7. `test_main_unknown_step_renders_progress_without_step_body[0]`
**Mutation:** Unknown step 0 falls into step-1 renderer

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_unknown_step_renders_progress_without_step_body[0]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:162: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:41.560 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_unknown_step_renders_progress_without_step_body[0]
1 failed in 0.89s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.83s
```

## 8. `test_main_unknown_step_renders_progress_without_step_body[6]`
**Mutation:** Unknown step 6 falls into step-1 renderer

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_unknown_step_renders_progress_without_step_body[6]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:162: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:43.945 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_unknown_step_renders_progress_without_step_body[6]
1 failed in 0.91s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.82s
```

## 9. `test_main_unknown_step_renders_progress_without_step_body[99]`
**Mutation:** Unknown step 99 falls into step-1 renderer

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_unknown_step_renders_progress_without_step_body[99]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:162: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:46.290 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_unknown_step_renders_progress_without_step_body[99]
1 failed in 0.91s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.86s
```

## 10. `test_main_step_buttons_route_to_adjacent_steps[1-\u27a1\ufe0f Next: Capital Allocation-2]`
**Mutation:** Step-1 Next writes wizard_step 99

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[1-\u27a1\ufe0f Next: Capital Allocation-2]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:192: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:48.761 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[1-\u27a1\ufe0f Next: Capital Allocation-2]
1 failed in 0.96s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.84s
```

## 11. `test_main_step_buttons_route_to_adjacent_steps[2-\u2b05\ufe0f Back: Analysis Mode-1]`
**Mutation:** Step-2 Back writes wizard_step 99

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[2-\u2b05\ufe0f Back: Analysis Mode-1]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:192: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:51.225 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[2-\u2b05\ufe0f Back: Analysis Mode-1]
1 failed in 0.93s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.84s
```

## 12. `test_main_step_buttons_route_to_adjacent_steps[2-\u27a1\ufe0f Next: Return & Risk-3]`
**Mutation:** Step-2 Next writes wizard_step 99

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[2-\u27a1\ufe0f Next: Return & Risk-3]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:192: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:53.583 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[2-\u27a1\ufe0f Next: Return & Risk-3]
1 failed in 0.91s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.82s
```

## 13. `test_main_step_buttons_route_to_adjacent_steps[3-\u2b05\ufe0f Back: Capital Allocation-2]`
**Mutation:** Step-3 Back writes wizard_step 99

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[3-\u2b05\ufe0f Back: Capital Allocation-2]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:192: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:55.941 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[3-\u2b05\ufe0f Back: Capital Allocation-2]
1 failed in 0.89s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.82s
```

## 14. `test_main_step_buttons_route_to_adjacent_steps[3-\u27a1\ufe0f Next: Correlations-4]`
**Mutation:** Step-3 Next writes wizard_step 99

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[3-\u27a1\ufe0f Next: Correlations-4]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:192: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:35:58.300 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[3-\u27a1\ufe0f Next: Correlations-4]
1 failed in 0.93s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.88s
```

## 15. `test_main_step_buttons_route_to_adjacent_steps[4-\u2b05\ufe0f Back: Return & Risk-3]`
**Mutation:** Step-4 Back writes wizard_step 99

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[4-\u2b05\ufe0f Back: Return & Risk-3]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:192: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:36:00.786 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[4-\u2b05\ufe0f Back: Return & Risk-3]
1 failed in 0.93s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.83s
```

## 16. `test_main_step_buttons_route_to_adjacent_steps[4-\u27a1\ufe0f Next: Review & Run-5]`
**Mutation:** Step-4 Next writes wizard_step 99

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[4-\u27a1\ufe0f Next: Review & Run-5]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:192: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:36:03.127 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[4-\u27a1\ufe0f Next: Review & Run-5]
1 failed in 0.90s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.83s
```

## 17. `test_main_step_buttons_route_to_adjacent_steps[5-\u2b05\ufe0f Back: Correlations-4]`
**Mutation:** Step-5 Back writes wizard_step 99

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[5-\u2b05\ufe0f Back: Correlations-4]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:192: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:36:05.478 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_buttons_route_to_adjacent_steps[5-\u2b05\ufe0f Back: Correlations-4]
1 failed in 0.91s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.83s
```

## 18. `test_main_sidebar_reset_routes_to_step_1[2]`
**Mutation:** Reset button writes wizard_step 3

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_sidebar_reset_routes_to_step_1[2]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:207: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:36:07.850 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_sidebar_reset_routes_to_step_1[2]
1 failed in 0.90s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.89s
```

## 19. `test_main_sidebar_reset_routes_to_step_1[3]`
**Mutation:** Reset button writes wizard_step 3

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_sidebar_reset_routes_to_step_1[3]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:207: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:36:10.396 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_sidebar_reset_routes_to_step_1[3]
1 failed in 0.97s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.92s
```

## 20. `test_main_sidebar_reset_routes_to_step_1[4]`
**Mutation:** Reset button writes wizard_step 3

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_sidebar_reset_routes_to_step_1[4]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:207: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:36:13.049 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_sidebar_reset_routes_to_step_1[4]
1 failed in 0.95s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.90s
```

## 21. `test_main_sidebar_reset_routes_to_step_1[5]`
**Mutation:** Reset button writes wizard_step 3

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_sidebar_reset_routes_to_step_1[5]
```

**RED tail:**
```
tests/test_coverage_round_2413.py:207: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:36:15.565 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_sidebar_reset_routes_to_step_1[5]
1 failed in 0.92s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.91s
```

## 22. `test_main_step_5_run_gate_opens_index_upload_without_executing`
**Mutation:** Missing-index run gate invokes pa_cli.main unconditionally

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_5_run_gate_opens_index_upload_without_executing
```

**RED tail:**
```
tests/test_coverage_round_2413.py:224: AssertionError
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:36:18.144 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_5_run_gate_opens_index_upload_without_executing
1 failed in 0.97s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.87s
```

## 23. `test_main_step_5_run_gate_stops_when_loaded_config_is_invalid`
**Mutation:** Invalid YAML no longer stops execution

**RED command:**
```
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2413.py::test_main_step_5_run_gate_stops_when_loaded_config_is_invalid
```

**RED tail:**
```
tests/test_coverage_round_2413.py:245: Failed
----------------------------- Captured stderr call -----------------------------
2026-11-03 12:36:20.706 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager
------------------------------ Captured log call -------------------------------
WARNING  streamlit.runtime.caching.cache_data_api:cache_data_api.py:372 No runtime found, using MemoryCacheStorageManager
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2413.py::test_main_step_5_run_gate_stops_when_loaded_config_is_invalid
1 failed in 0.98s
```

**GREEN after revert:**
```
.                                                                        [100%]
1 passed in 0.87s
```
