# Retained coverage and failure recheck for #2477

**13,493 / 14,626 statements covered (92.25352112676056%).** These are the exact
totals in [coverage.json](coverage.json), reconciled against every module by
[summary.json](summary.json). The completed routine suite reports **2,008 passed,
2 failed, 3 skipped, 4 deselected**, with no errors. Pytest finishes in 175.27 seconds
(178.01 seconds including process startup). The outer limit is 600 seconds with a
30-second termination grace.

[full-suite.txt](full-suite.txt) retains the console output and
[full-suite.xml](full-suite.xml) retains named-node outcomes. The exact argv,
working directory, interpreter, base revision, input hashes and exit status are
in [full-suite-command.json](full-suite-command.json). The production base is main
`7613ffd97a450b15e550f499b7386ec4e67f2c51`; the starting PR head is
`e3605c43efddc59fb3a8f207729fb58d96f1752c`. Production and coverage configuration
have no diff against that main revision. This is a measurement of current-main
production with PR test changes, rather than a claim that the entire test tree
equals main.

```sh
timeout --signal=INT --kill-after=30s 600s env COVERAGE_FILE=/tmp/issue2336-recheck.coverage python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:docs/evidence/issue-2336-png-failures/recheck-20261104/coverage.json --junitxml=docs/evidence/issue-2336-png-failures/recheck-20261104/full-suite.xml -m 'not slow and not live_llm' -o faulthandler_timeout=60 tests/
```

The inherited test change bounds the native PNG CLI child process at 30 seconds;
timeouts remain failures. This round adds six evidence-manifest cases and a second
optimizer-import fallback case, accounting for the seven additional passing tests
over the prior capture. The keepalive instructions require exclusion of slow
tests, and the existing default excludes live providers. The four deselected nodes
are the wheel-install test below, the two live smoke tests in
`tests/test_llm_live_smoke.py`, and
`tests/test_tracing.py::test_authorized_synthetic_trace_live_retrieval`.
This is not an all-tests PASS or a successful native PNG export.

## Failure dispositions

| Test | Evidence and disposition |
| --- | --- |
| `tests/test_cli.py::test_main_with_png` | Fails after the inherited 30-second subprocess bound. A fresh Chromium launch in [browser-probe.txt](browser-probe.txt) aborts with `setsockopt: Operation not permitted` and a crash-handler startup failure. Environmental browser restriction, independently reproduced outside pytest. The original unbounded test-harness defect was repaired in the previous commit. A successful native rendering run remains outstanding. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | Fails before package installation: its isolated interpreter cannot resolve `pypi.org` to install setuptools. The DNS errors are retained in the full-suite console output. Environmental network restriction. In PR #2475 the same test instead reached installation and was rejected because Python 3.12.2 does not satisfy the package's `>=3.13` requirement. |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | One of PR #2475's four failures; deselected here by the required `not slow` marker. The historical wheel subprocess ran under unsupported Python 3.12, but its captured stderr was not retained, so the exact wheel-build cause remains unproven. A supported-Python wheel run is still required. |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | PR #2475 fails with Python 3.12's `Optional[Dict[...]]` representation instead of the committed `Dict[...] \| None`. Environmental unsupported-interpreter difference. It passes unchanged under Python 3.14.9 in both the full suite and the focused regressions. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | Historical `assert False` does not retain `optimizer_status`; the historical log also warns about SciPy's `np.long` use. An incompatible optional optimizer is plausible, but not conclusively established. The current test skips because SciPy is absent. Installing SciPy from PyPI was attempted and returned no available distribution in this environment; see [scipy-install.txt](scipy-install.txt). Do not classify the historical assertion as definitively environmental or close #2336 yet. |

The remaining two skips are the Anthropic constructor option cases in
`tests/test_dashboard_llm_settings.py` and `tests/test_llm_provider_missing_keys.py`;
both require the absent optional `langchain_anthropic` package. Named outcomes and
reasons are preserved in the JUnit report.

The optimizer regression now exercises the real import and suggestion path for
both `ImportError` and `AttributeError`, requiring feasible grid results,
`optimizer_success=False`, and `grid_fallback:missing_scipy` on every row.
The real-SciPy success assertion now includes the result row so future failures
retain the diagnostic status. These tests do not substitute for a compatible
SciPy/NumPy integration run.

The largest gaps remain `pa_core/cli.py` (165 statements),
`pa_core/sleeve_suggestor.py` (115), `dashboard/pages/3_Scenario_Wizard.py` (101),
`dashboard/pages/5_Scenario_Grid.py` (57), and `pa_core/sim/paths.py` (42).
All module totals and missing lines are in the summary. Coverage exceeds 90%, so
the conditional below-target module addition is not needed.

## Evidence retention and PNG proof

The previous commit omitted nine `.log` files named in its manifest because Git
ignores that extension. [previous-capture-audit.json](previous-capture-audit.json)
names all nine and verifies the 17 artifacts that remain. This round's captures
use `.txt`; no unavailable original log is represented as recovered. The new
`summarize.py --manifest` option rejects a missing, truncated or altered declared
artifact before writing a report. Six new tests exercise these checks and the CLI.

One historical log was recoverable: [pr2475-full-suite.txt](pr2475-full-suite.txt)
is the byte-identical `final-candidate.log` from commit
`d112c2238441e3f9035f42360f5cf1e51839f5ea`'s `current-validation.tar.gz`. Its SHA256
is `d28f3eb0664197015fad24c3ff8600e168d838a8c15e67c7a5c227e3a0ecd12e`, matching
the previous manifest. The archive hash is
`7b8f6122b0e69e310693676459e886b7c6a9e1014c6bc114198fe516a8583e91`.
The existing [reconciliation](../main-20261104/pr2475-reconciliation.json) establishes
that PR #2475's JSON has **13,537 / 14,670 statements (92.27675528289025%)**;
13,441 was a prose error of 96 statements. The historical and current toolchain
totals remain separate.

The fresh focused PNG/PDF/sweep command passes **23 tests**, with argv, console and
JUnit retained here. [png-audit.json](png-audit.json) independently verifies all
41 members of the prior PNG archive against their recorded sizes and SHA256
hashes, then checks each of the eight exact-node AssertionError RED/restored GREEN
pairs with the strict JUnit validator. Original, restored and GREEN production
hashes all equal current main's CLI SHA256
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
The archive remains unchanged; no additional mutation execution is claimed here.
PDF and next-boundaries evidence is untouched.

The focused evidence, schema and sleeve regressions report **58 passed, 1 skipped**.
Targeted tooling coverage reports `summarize.py` at **78%**; this is separate from
product coverage, and no tooling coverage threshold is claimed.
The required repository-wide Black command passes for **534 Python files**;
[black-command.json](black-command.json) records the command and temporary-cache
workaround for the sandbox's multiprocessing/thread-pool restrictions. Ruff and
`git diff --check` pass for source, JSON and Markdown changes. Raw console and
JUnit captures preserve pytest's original trailing whitespace.

## Reconciled task status

- [x] Completed bounded routine-suite measurement with retained console, JUnit,
  raw coverage JSON, exact totals, command, revision and module gaps.
- [ ] Conclusively disposition all four historical failures: compatible SciPy
  and supported-Python wheel verification remain outstanding.
- [x] Verify the previous focused run and eight PNG mutation pairs against main;
  repeat the focused run with durable console output.
- [ ] Publish the evidence comment and close #2336 after the remaining historical
  dispositions are established.
- [x] README covered/total statements equal the raw coverage JSON exactly.
- [x] Every failure in this completed run is named with supporting evidence.
- [x] Measured product coverage exceeds 90%.

Updating the PR body was attempted after verifying the prior evidence, but the
GitHub tool rejected it: `MCP tool call requires approval, but approval policy is
never`. The reconciled statuses above are local; the remote checkboxes were not
changed. PR #2478 was open and ready for review when inspected. No issue was closed.
The checkout's `.git` is read-only, and even `git add --dry-run` cannot create
`index.lock`. The delivery commit uses temporary Git metadata at
`/tmp/issue2336-delivery.git`, leaving the checkout edits available for automation.
It does not advance this checkout's branch or publish changes to GitHub.
