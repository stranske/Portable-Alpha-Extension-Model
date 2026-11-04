# Main coverage with configuration drift rejected

**13,493 / 14,626 covered statements (92.25352112676056%).** These counts and
the percentage equal [main-capture/coverage.json](main-capture/coverage.json)
exactly. [main-capture/summary.json](main-capture/summary.json) reconciles every
module and missing line and ranks the 15 largest source gaps.
[capture-summary.txt](capture-summary.txt) preserves the printed percentage.

The completed run reports **1,988 passed, 2 failed, 3 skipped, 4 deselected**,
exit 1, in 196.51 pytest seconds (199.70 seconds including startup and reporting).
Raw console and named-node JUnit are retained in
[main-capture/full-suite.txt](main-capture/full-suite.txt) and
[main-capture/full-suite.xml](main-capture/full-suite.xml).

## Measured tree and reproduction

GitHub main was verified as `06881dbdf0e915c305669e7fe3f8baa7d8a21ec4`.
The run uses a separate local clone checked out at that exact commit, with
unchanged product files and configuration. Its only tracked difference is the
existing 30-second native PNG subprocess bound, retained in
[main-capture/tests.patch](main-capture/tests.patch). Candidate tests, including
this round's new regressions, do not contribute to this main-suite pass count.

[main-capture/command.json](main-capture/command.json) records the actual argv,
revision, interpreter, tool versions, coverage data location and input hashes.
The interpreter is Python 3.14.10, with pytest 9.2.0, pytest-cov 7.1.0 and coverage
7.17.0. Main pins coverage 7.17.1; the actual installed version is disclosed rather
than presented as matching that pin.

The capture helper runs this command, using absolute output paths:

```sh
timeout --signal=INT --kill-after=10s 600s python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:coverage.json --junitxml=full-suite.xml -m 'not slow and not live_llm' tests/
```

The required slow-test exclusion and existing live-provider exclusion deselect
the online wheel installation test and three provider tests. This is a completed
routine suite with a documented test overlay, not an untouched all-tests PASS.

## Capture repair and verification

The helper previously accepted changed test/coverage configuration while claiming
to measure main. It now rejects modified, deleted, staged or newly added root
configuration files before creating evidence or starting pytest. This includes
pytest's dedicated TOML files and gitignored configuration: Git's ignore rules
do not stop pytest or coverage from reading a local configuration file.

Ten new regressions exercise these rejection paths. The original seven fail in
[config-red.txt](config-red.txt); three additional TOML/ignored-file cases fail
against the intermediate repair in [config-extra-red.txt](config-extra-red.txt).
All **69 capture/summary/replay regressions pass** in
[regressions.txt](regressions.txt) and [regressions.xml](regressions.xml), including
the successful capture paths and a real stalled subprocess. The tooling coverage
table is separate from the product coverage measurement above; no tooling
coverage threshold is claimed. All **23 focused PNG/PDF/sweep tests pass** in
[focused.txt](focused.txt).

The measured helper source is preserved as
[main-capture/capture-input.txt](main-capture/capture-input.txt), with bytes equal
to `capture_sha256` in the run metadata. The final helper adds the TOML and
gitignored-file checks after that run; those checks execute before pytest and
do not alter the measured test command or inputs. [verification.json](verification.json)
records the source/input comparisons and validation commands. The required
whole-tree Black check passes for 536 files; Ruff passes. Black's single-file
checks prime a writable cache because the sandbox denies multiprocessing IPC.

| Largest source gap | Missing statements |
| --- | ---: |
| `pa_core/cli.py` | 165 |
| `pa_core/sleeve_suggestor.py` | 115 |
| `dashboard/pages/3_Scenario_Wizard.py` | 101 |
| `dashboard/pages/5_Scenario_Grid.py` | 57 |
| `pa_core/sim/paths.py` | 42 |

## Failures and historical limits

| Node | Cause or disposition |
| --- | --- |
| `tests/test_cli.py::test_main_with_png` | Current run exceeds the retained 30-second subprocess bound. The previous independent [browser probe](../main-06881dbd-20261104/browser-probe.txt) establishes Chromium startup failure with `setsockopt: Operation not permitted` in this sandbox. Environmental rendering restriction; this run is not a native PNG export witness. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | Current run cannot resolve `pypi.org` while installing setuptools. Console/JUnit retain the DNS errors. Environmental network restriction before entrypoint behavior is reached. PR #2475 instead used unsupported Python 3.12.2 and failed the package's `>=3.13` requirement. |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | PR #2475's unsupported Python 3.12 rendered the schema type text differently. The unchanged test passes in the supported-Python main capture. |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | PR #2475 did not preserve the wheel-build stderr, so the exact historical cause remains unresolved. This slow online test is deselected here. Prior supported-Python offline installation evidence remains unchanged. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | PR #2475 retained an assertion failure and a SciPy/NumPy warning, insufficient to establish the cause. Current main skips this test because SciPy is absent. Compatible optimizer integration remains outstanding. |

The other two skips require absent `langchain_anthropic`. Historical PR #2475's
raw JSON remains **13,537 / 14,670 (92.27675528289025%)**; its prose numerator
13,441 was incorrect. The previous [reconciliation](../main-20261104/pr2475-reconciliation.json)
and historical evidence are untouched.

[retained-png-audit.json](retained-png-audit.json) independently verifies all eight
retained assertion RED/restored GREEN pairs at the same current main revision.
Every original/restored/GREEN CLI hash equals current main's
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
This is validation of the existing same-main receipts, not a fresh mutation run.
PDF and next-boundaries evidence are untouched.

- [x] Completed bounded main coverage capture, with counts, module totals and gaps.
- [x] README counts match raw coverage JSON and its printed percentage exactly.
- [x] Every current failing test is named with a cause or environmental disposition.
- [x] Total product coverage exceeds 90%.
- [ ] Conclusively disposition the historical optimizer and wheel-build failures.
- [ ] Publish final acceptance evidence and close #2336 after those dispositions.

#2336 remains open. The checkout's `.git` is read-only, so the change is committed
in a separate writable clone and preserved in `/tmp/issue2336-config-guard.bundle`
and `/tmp/issue2336-config-guard.patch`. This does not advance the checkout branch
or publish the commit. `/tmp/issue2336-config-guard-delivery.json` records the
commit and artifact identities for handoff.

Automatic approval review rejected the PR checklist update with
`MCP tool call requires approval, but approval policy is never`. The verified
14/18 checklist is preserved in `/tmp/issue2336-config-guard-pr-body.md`; the
remote PR body remains unchanged. PR #2478 is open and ready for review.
