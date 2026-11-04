# Verified coverage capture for #2477 / #2336

**13,493 / 14,626 statements covered (92.25352112676056%).** These figures equal
[coverage.json](coverage.json) exactly. [summary.json](summary.json) reconciles
every module, missing line, aggregate count and percentage, the 15 largest source
gaps, and named failing tests. [summary.txt](summary.txt) preserves the printed
percentage. No product code or coverage policy changed.

GitHub main was verified through the GitHub connector on 2026-11-04 as
`7613ffd97a450b15e550f499b7386ec4e67f2c51`. The capture starts from branch head
`9ff81e446e598f118a04363cb7072f65955e1f87`, with the seven new evidence-helper
regressions and helper repair described below. `pa_core/`, `dashboard/` and
`pyproject.toml` have no differences from main. The exact argv, source hashes,
interpreter and dependency versions are in [full-suite-command.json](full-suite-command.json).

The completed suite reports **2,018 passed, 2 failed, 3 skipped, 4 deselected**,
exits 1 and takes 210.52 pytest seconds (214.20 seconds including process startup
and shutdown). [full-suite.txt](full-suite.txt), [full-suite.xml](full-suite.xml)
and [full-suite.exit](full-suite.exit) retain the console, named-node JUnit and exit.

```sh
timeout --signal=INT --kill-after=30s 600s env COVERAGE_FILE=/tmp/issue2336-verified.coverage python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:docs/evidence/issue-2336-png-failures/verified-20261104/coverage.json --junitxml=docs/evidence/issue-2336-png-failures/verified-20261104/full-suite.xml -m 'not slow and not live_llm' -o faulthandler_timeout=60 tests/
```

This is the completed routine suite with the required `not slow` selection and
the repository's default live-provider exclusion. It includes the inherited
30-second native-PNG subprocess bound and offline installed-wheel check. It is
a measurement of current-main production with documented test changes, not an
execution of the untouched main test tree or an all-tests PASS. The excluded
tests are the online wheel-install test and three live-provider tests.

## Failures, skips and remaining historical uncertainty

| Node | Cause or disposition |
| --- | --- |
| `tests/test_cli.py::test_main_with_png` | `subprocess.TimeoutExpired` after 30 seconds. The native renderer still cannot finish in this sandbox. The retained [browser probe](../recheck-20261104/browser-probe.txt) establishes Chromium's `setsockopt: Operation not permitted` startup failure in this environment. Environmental rendering restriction; a native export witness in a working browser environment remains outstanding. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | pip cannot resolve `pypi.org` while installing setuptools; captured stderr names the DNS error. Environmental network restriction, before entrypoint behavior is reached. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | Skipped because SciPy is absent. This does not disposition PR #2475's optimizer assertion; the historical status was not retained and compatible SciPy/NumPy integration remains necessary. |
| `tests/test_dashboard_llm_settings.py::test_audit_provider_options_reach_real_offline_constructor[anthropic]` | Skipped because `langchain_anthropic` is absent. |
| `tests/test_llm_provider_missing_keys.py::test_create_llm_provider_options_keep_supported_fields_out_of_anthropic_payload` | Skipped because `langchain_anthropic` is absent. |

The [original dispositions](../main-20261104/README.md#failure-dispositions)
name all four PR #2475 failures as well. Its schema representation difference
passes unchanged under supported Python. The current offline wheel installation
passes in this full run, but cannot recover the historical wheel-build stderr.
That precise historical cause and the optimizer cause remain unresolved; #2336
must remain open. The archived PR #2475 JSON establishes 13,537 / 14,670 covered
statements; 13,441 was a prose transcription error, not this capture's numerator.

Largest remaining gaps: `pa_core/cli.py` (165 missing statements),
`pa_core/sleeve_suggestor.py` (115), and
`dashboard/pages/3_Scenario_Wizard.py` (101). Coverage exceeds 90%, so the
conditional below-target module addition is unnecessary. A future focused
product-coverage round can use `pa_core/sleeve_suggestor.py` and
`tests/test_sleeve_suggestor.py` after resolving its optional dependency.

## Verified helper repair and retained PNG evidence

The helper previously accepted a percentage inconsistent with its statement
counts, including non-finite percentages. It now checks each module and the
aggregate against covered/total statements, with an absolute floating-point
tolerance of `1e-12`. Empty modules retain coverage.py's 100% convention.
The six rejection cases fail before the fix; all **26 helper tests pass** after
the fix, including the empty-module regression. [checks.json](checks.json)
records commands and outcomes, [percentage-red.txt](percentage-red.txt) and
[percentage-green.txt](percentage-green.txt) retain results, and the GREEN
[JUnit](percentage-green.xml) names every helper case. The separately measured
80% helper coverage is tooling coverage, outside the product percentage.

The required whole-tree Black check passes for 534 Python files after individual
cache priming to avoid sandbox multiprocessing restrictions. Ruff passes for
both changed Python files. The first Black check timed out while priming was
still running; both that attempt and the successful final check are retained.

[png-audit.json](png-audit.json) revalidates all 41 retained PNG archive members,
eight exact-node assertion RED/restored GREEN pairs, and unchanged current test
and production hashes. Every restored CLI hash equals current main's
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
This is an audit of the already executed pairs, not a new mutation run.
The historical PDF and next-boundaries evidence is untouched.

## Reconciled task status

- [x] Completed bounded current-main production coverage capture, with counts,
  per-module totals, largest gaps, command and Git revision.
- [x] README totals equal the raw coverage JSON and percentages are validated.
- [x] Previously committed current-main PNG pairs still match source/test hashes
  and each has the required assertion RED and restored GREEN outcome.
- [x] Every current failing or timed-out test is named above with its cause.
- [x] Product coverage exceeds 90%.
- [ ] Conclusively disposition the historical optimizer and wheel-build failures.
- [ ] Publish final evidence and close #2336 after those dispositions.

Delivery and remote-checklist results are recorded in `delivery.json`.
The checkout's Git metadata is read-only; an independently reviewable delivery
commit can be retained under `/tmp` without advancing this checkout's branch.
