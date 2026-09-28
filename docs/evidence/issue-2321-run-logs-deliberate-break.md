# Issue 2321: Run Logs manifest provenance deliberate-break evidence

This is a current reproduction on `main` at `75ad1fbd48e9f0e7b30e5c3b92a9a5778683c9e7`. It supplies the missing falsification evidence for issue #2284 and merged PR #2315 (`8b984917b7c893ecbf599b725600a09495e57bc8`). It is not represented as historical output from the original PR.

## Gate

The production page in `dashboard/pages/7_Run_Logs.py` must show only the manifest named by the selected run's `run_end.json`. It must not fall back to an unrelated `manifest.json` in the current working directory.

The named regression is:

```text
uv run pytest tests/test_dashboard_run_logs_page.py::test_run_logs_ignores_unrelated_root_manifest -q
```

The repository's default `uv run` environment does not register pytest-cov's `--no-cov` option, so the executable repository-supported command above intentionally omits that unsupported flag.

## Deliberate break

The break restored only the historical fallback removed by PR #2315:

```python
from pa_core.contracts import MANIFEST_FILENAME

if found_manifest is None or not found_manifest.exists():
    for cand in Path.cwd().glob(MANIFEST_FILENAME):
        found_manifest = cand
        break
```

With that mutation present, the named test failed because the page rendered the unrelated root manifest:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
________________ test_run_logs_ignores_unrelated_root_manifest _________________

>       assert not any("WRONG_UNRELATED_RUN" in payload for payload in code_payloads)
E       assert not True
E        +  where True = any(<generator object ...>)

tests/test_dashboard_run_logs_page.py:143: AssertionError
=========================== short test summary info ============================
FAILED tests/test_dashboard_run_logs_page.py::test_run_logs_ignores_unrelated_root_manifest
1 failed in 1.37s
```

## Restoration

After removing the fallback and the temporary import, the exact named test passed:

```text
.                                                                        [100%]
1 passed in 1.35s
```

The complete focused file also passed:

```text
$ uv run pytest tests/test_dashboard_run_logs_page.py -q
.......                                                                  [100%]
7 passed in 1.33s
```

`git diff -- dashboard/pages/7_Run_Logs.py tests/test_dashboard_run_logs_page.py` was empty after restoration, so no deliberate-break mutation remains in the deliverable.
