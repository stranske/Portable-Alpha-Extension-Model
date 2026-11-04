# Reject inherited test options before capturing main coverage

**13,493 / 14,626 covered statements (92.25352112676056%).** These values equal
[main-capture/coverage.json](main-capture/coverage.json) exactly. The percentage
is printed in [capture-summary.txt](capture-summary.txt). The completed run has
**1,988 passed, 2 failed, 3 skipped, 4 deselected**, exit 1, in 141.28 pytest
seconds (143.58 seconds including startup and reporting).

[main-capture/summary.json](main-capture/summary.json) reconciles every module,
its statement counts, missing line numbers and the 15 largest source gaps.
[main-capture/full-suite.txt](main-capture/full-suite.txt) and
[main-capture/full-suite.xml](main-capture/full-suite.xml) retain the raw console
and named-node JUnit. This is a completed routine suite with environmental
failures; it does not establish a native PNG export success.

## Measured source and command

GitHub main was verified as `06881dbdf0e915c305669e7fe3f8baa7d8a21ec4` through the
GitHub connector. A separate local clone checks out that exact commit. Its
product files and configuration match main. Its only tracked difference is the
existing 30-second native PNG subprocess bound, retained in
[main-capture/tests.patch](main-capture/tests.patch). Candidate tests do not
contribute to this main-suite pass count.

[main-capture/command.json](main-capture/command.json) records the actual absolute
argv, revision, interpreter, tool versions and input hashes. The command is:

```sh
timeout --signal=INT --kill-after=10s 600s python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:coverage.json --junitxml=full-suite.xml -m 'not slow and not live_llm' tests/
```

The required slow exclusion and existing live-provider exclusion deselect the
online wheel installation test and three provider tests. The interpreter is
Python 3.14.8, with pytest 9.2.0, pytest-cov 7.1.0 and coverage 7.17.0. Main pins
coverage 7.17.1; this capture discloses the installed version. The measured
helper is preserved byte-for-byte as
[main-capture/capture-input.txt](main-capture/capture-input.txt).

| Largest source gap | Missing statements |
| --- | ---: |
| `pa_core/cli.py` | 165 |
| `pa_core/sleeve_suggestor.py` | 115 |
| `dashboard/pages/3_Scenario_Wizard.py` | 101 |
| `dashboard/pages/5_Scenario_Grid.py` | 57 |
| `pa_core/sim/paths.py` | 42 |

## Capture defect and repair

The helper previously inherited `PYTEST_ADDOPTS`, `COVERAGE_RCFILE` and
`COVERAGE_PROCESS_START` without checking them. For example, `PYTEST_ADDOPTS=-x`
can end the suite at its first failure with exit 1 and internally consistent
partial JUnit and coverage reports. Those reports pass the existing summary
checks, so unchanged source/configuration files alone did not establish that
the documented full-suite command actually ran.

The helper now rejects nonempty overrides before creating evidence or starting
pytest, naming only the variable. It passes a snapshot of the checked environment
to the child. Three new parameter cases all fail against the old helper in
[environment-red.txt](environment-red.txt) and pass with the repair. All **72
capture/summary/replay regressions pass** in [regressions.txt](regressions.txt)
and [regressions.xml](regressions.xml), including successful captures and a real
stalled subprocess. Tooling coverage is reported separately from product
coverage; no tooling percentage target is claimed. The required whole-tree
Black check passes for 536 files; Ruff passes. Serial checks primed a writable
Black cache because worker IPC is restricted in this sandbox.

## Fresh PNG replay

All **23 focused PNG/PDF/sweep tests pass** in [focused.txt](focused.txt). A
fresh execution of `replay.py --revision 06881dbdf0e915c305669e7fe3f8baa7d8a21ec4`
produces eight assertion RED/restored GREEN pairs. [png-proof/results.json](png-proof/results.json)
records every exact node, argv, exit and original/mutant/restored hash. The
individual console logs and named-node JUnit files are retained in `png-proof/`.
Mutant sources are individually gzip-compressed; decompressed bytes match their
recorded hashes. [png-audit.json](png-audit.json) records an independent validation
of every pair and named JUnit class.

Every original, restored and GREEN CLI hash equals current main's
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
Production files in the caller remain unchanged. PDF and next-boundaries
evidence remain untouched.

## Current and historical failures

| Test | Cause or remaining uncertainty |
| --- | --- |
| `tests/test_cli.py::test_main_with_png` | Current main capture exceeds the retained 30-second subprocess bound. The retained [browser probe](../main-06881dbd-20261104/browser-probe.txt) independently establishes Chromium startup restrictions (`setsockopt: Operation not permitted`). Environmental; this run does not establish successful rendering. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | Current main capture fails DNS resolution for `pypi.org` while installing setuptools, before entrypoint behavior. Environmental. The candidate's previously verified offline repair is absent from this main-only capture. Historical PR #2475 used unsupported Python 3.12.2, failing the package's Python >=3.13 requirement. |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | Historical unsupported Python 3.12 changed the rendered schema type text. The unchanged test passes in this supported-Python main capture. Environmental. |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | Historical build stderr was discarded, leaving the exact cause unresolved. The slow online test is deselected as required; earlier supported-Python offline wheel installation evidence remains valid. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | Historical assertion and SciPy/NumPy warning do not establish import incompatibility versus solver failure. Current main skips because SciPy is absent. A fresh bounded attempt to install SciPy found no available distribution in this environment; [scipy-install.txt](scipy-install.txt) retains that limitation. Compatible-SciPy execution remains unavailable. |

The other two skips require absent `langchain_anthropic`. PR #2475's historical
JSON remains **13,537 / 14,670 (92.27675528289025%)**; its prose numerator 13,441
was incorrect by 96 statements. The earlier
[reconciliation](../main-20261104/pr2475-reconciliation.json) and historical
evidence remain unchanged.

## Verified checklist

- [x] Completed bounded current-main coverage capture, with counts, every module and gaps.
- [ ] Conclusively disposition all four historical failures and the PNG timeout.
- [x] Fresh 23-test export run and eight assertion RED/restored GREEN PNG pairs.
- [ ] Publish final evidence and close #2336 after historical dispositions are conclusive.
- [x] Retain completed raw coverage JSON, exact percentage, command and main SHA.
- [x] README numerator and denominator equal coverage JSON exactly.
- [x] Restored PNG production hashes equal current main's CLI hash.
- [x] Every failure in this completed main run has an environmental cause recorded above.
- [x] Total product coverage exceeds 90%.

These statuses also apply to the repeated tasks and acceptance criteria: **14/18
verified**, with the historical disposition and conditional closure tasks open.
No unresolved historical failure is marked environmental merely because its
dependency is unavailable here. A final GitHub connector check found #2336
already closed. This run did not change its state or establish the two missing
historical dispositions.

The checkout's `.git` is read-only: `git add --dry-run` cannot create `index.lock`.
The source, tests and evidence are therefore committed in a separate writable
clone, with a Git bundle and patch under `/tmp/issue2336-environment-guard.*`.
That commit does not advance the checkout branch or publish it to GitHub.
