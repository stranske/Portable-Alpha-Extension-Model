# Current main coverage and fresh PNG mutation receipts

**13,493 / 14,626 covered statements (92.25352112676056%).** These are the exact
totals in [coverage.json](coverage.json), independently reconciled per module in
[summary.json](summary.json). The completed run reports **1,988 passed, 2 failed,
3 skipped, 4 deselected**, with no errors. It finishes in 194.57 seconds
(198.08 seconds including startup), below the 600-second outer limit.

## Measured tree and command

Main was verified through GitHub as
`06881dbdf0e915c305669e7fe3f8baa7d8a21ec4`. This is newer than the earlier captures'
`7613ffd9` base: the intervening commit updates the development coverage pin.
Production is byte-identical between these revisions and the active PR checkout.
This capture uses an isolated Git checkout of the newer main, including its tests
and dependency configuration. Its only tracked difference is the inherited
30-second PNG subprocess bound in `tests/test_cli.py`, preserved in
[bounded-png-test.patch](bounded-png-test.patch). The PR's additional tests are
measured separately and do not inflate this main-suite pass count.

[full-suite-command.json](full-suite-command.json) records exact argv, checkout
revision/status, interpreter, versions, snapshot archive hash and overlay hash.
[full-suite.txt](full-suite.txt) and [full-suite.xml](full-suite.xml) retain console
output and named-node outcomes. Installed coverage is **7.17.0**, while the new
main development pin is **7.17.1**; this records the actual toolchain rather than
claiming to have installed the new pin. Python is 3.14.9 and NumPy is 2.5.4.

```sh
timeout --signal=INT --kill-after=30s 600s python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:coverage.json --junitxml=full-suite.xml -m 'not slow and not live_llm' -o faulthandler_timeout=60 tests/
```

The recorded command uses absolute output paths into this evidence directory.
The required `not slow` marker and existing default exclusion of live providers
deselect the slow online wheel test and three live-provider tests. The skipped
tests require absent SciPy or `langchain_anthropic`. This is a completed routine
suite, not an all-tests PASS or a successful native PNG render.

The initial snapshot lacked Git metadata, causing
`tests/test_reference_packs.py::test_reference_directory_is_gitignored` to fail.
Its console, JUnit, command, coverage and summary are retained as `initial-*`
artifacts (the latter two are gzip-compressed). Adding private Git metadata fixed
that harness error: [snapshot-fix.txt](snapshot-fix.txt) records the single-node
pass, and the repeated full run above also passes that node. No product change
was required; the repeated run supplies the headline results.

## Failure dispositions

| Test | Cause and disposition |
| --- | --- |
| `tests/test_cli.py::test_main_with_png` | Current run fails at the 30-second native renderer bound. An independent [Chromium probe](browser-probe.txt) aborts with `setsockopt: Operation not permitted` and crash-handler startup failure, exit -6. Environmental sandbox restriction. [browser-command.json](browser-command.json) records argv and timeout. The bounded test repair preserves failure without hanging coverage collection. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | Current run cannot resolve PyPI while installing setuptools into its isolated interpreter. Environmental network restriction; console and JUnit retain DNS errors. PR #2475 instead reached package installation and explicitly failed because Python 3.12.2 does not satisfy `>=3.13`. |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | PR #2475's unsupported Python 3.12 rendered `Optional[Dict[...]]` differently from the committed `Dict[...] \| None`. Environmental interpreter difference. The unchanged test passes in this supported-Python full run. |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | PR #2475's wheel-build subprocess failed under unsupported Python 3.12, but its stderr was not retained. Its exact historical cause remains unresolved; this slow test is deselected here. The prior [supported-Python offline installed-wheel verification](../install-diagnostics-20261104/README.md) passes and preserves future subprocess diagnostics, but does not recover that cause. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | PR #2475 retained only `assert False` and a SciPy warning about `np.long`; incompatibility is plausible, not proven. Current execution skips because SciPy is absent. A fresh [binary wheel availability check](scipy-availability.txt) finds no distribution. Compatible SciPy/NumPy integration and the historical optimizer status remain unavailable. |

The initial snapshot-only repository-ignore failure is explained and fixed above.
The [retained historical console](../recheck-20261104/pr2475-full-suite.txt) names
all four PR #2475 failures. Its [JSON reconciliation](../main-20261104/pr2475-reconciliation.json)
establishes **13,537 / 14,670 (92.27675528289025%)** for that historical capture;
13,441 was a prose error of 96 statements. These historical totals are not mixed
with the new coverage JSON. The earlier [tool comparison](../main-20261104/coverage-tool-comparison.json)
records the 44-statement denominator difference in `pa_core/wizard_schema.py`
across the two Python/coverage toolchains.

Largest source gaps are `pa_core/cli.py` (165 statements),
`pa_core/sleeve_suggestor.py` (115), `dashboard/pages/3_Scenario_Wizard.py` (101),
`dashboard/pages/5_Scenario_Grid.py` (57), and `pa_core/sim/paths.py` (42).
The summary retains every module's totals and missing line numbers. Product
coverage exceeds 90%, so the conditional additional module task is unnecessary.

## Eight fresh PNG RED/GREEN pairs

The focused PNG/PDF/sweep command passes **23 tests**, recorded in
[focused.txt](focused.txt) and [focused.xml](focused.xml). [png-commands.json](png-commands.json)
records both that command and the new replay invocation, with argv, input hashes,
working directory, elapsed time and exit status.

```sh
python -m pytest tests/test_cli_png_failures.py tests/test_cli_pdf_failures.py tests/test_cli_sweep_png_export.py -q -m 'not slow and not live_llm'
python docs/evidence/issue-2336-png-failures/replay.py --revision 06881dbdf0e915c305669e7fe3f8baa7d8a21ec4 --output <new-proof-directory>
```

Each of `renderer`, `module`, `io`, `permission`, `value`, `type`,
`runtime-renderer` and `runtime-other` ran its exact named test node twice.
Every RED exits 1 with an AssertionError; every restored GREEN exits 0 without
failures, errors or skips. [png-proof/results.json](png-proof/results.json) records
each mutation anchor, command and source hashes. Each case's raw `*-red.txt`,
`*-red.xml`, `*-green.txt` and `*-green.xml` is readable directly in `png-proof/`.
The eight mutant sources are individually gzip-compressed after successful replay;
their decompressed hashes equal the recorded mutant hashes.

Every original/restored/GREEN CLI SHA256 equals current main's
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
[png-audit.json](png-audit.json) records independent exact-node JUnit and byte
validation. Production and the caller's boundary test remain unchanged. All
historical evidence, including PDF and next-boundaries evidence, is untouched.

The replay driver now accepts `--revision` (default `origin/main`), resolves it
once to an immutable commit, and refuses a caller CLI that differs from that
snapshot. Two regressions fail against the old driver in [driver-red.txt](driver-red.txt);
all **40 replay/summary regressions pass** in [driver-green.txt](driver-green.txt)
and [driver-green.xml](driver-green.xml). The required whole-tree
[Black check](black-check.txt) passes for all 534 Python files; Ruff also passes.
The initial sandbox multiprocessing failure and sequential cache priming are
retained separately. [checks.json](checks.json) records validation and delivery
constraints; [manifest.json](manifest.json) binds the retained evidence bytes.

## Reconciled checklist and remaining blockers

The prior six implementation commits were reviewed before continuing. These nine
statuses apply to both copies of the PR checklist: **14/18 verified**.

- [x] Completed bounded coverage capture on current-main production and tests,
  with counts, module totals, gaps, command and revision.
- [ ] Conclusively disposition every historical failure, including the optimizer
  and wheel-build failures whose diagnostic output is unavailable.
- [x] Fresh focused export run and eight assertion RED/restored GREEN pairs.
- [ ] Publish final evidence and close #2336 after historical dispositions.
- [x] Completed-run raw coverage JSON retained with percent, command and Git SHA.
- [x] README covered/total statements equal coverage JSON exactly.
- [x] Eight pairs have AssertionError RED, GREEN and byte-identical main CLI hashes.
- [x] Every current failing or timed-out test is named with its cause or fix.
- [x] Total product coverage exceeds 90%.

#2336 remains open. The historical optimizer and wheel-build classifications are
not established by current environmental failures or absent optional packages.

The PR-body reconciliation, `needs-human` label and evidence comment were rejected
because the GitHub tools require approval and approval policy is `never`. PR #2478
was verified open and ready for review after those attempts. The pending body is
`/tmp/issue2336-main-rebind-pr-body.md`. The checkout's Git metadata is read-only:
`git add --dry-run` cannot create `index.lock`. A separate delivery commit, patch
and bundle use `/tmp/issue2336-main-rebind.git`,
`/tmp/issue2336-main-rebind.patch` and `/tmp/issue2336-main-rebind.bundle`;
`/tmp/issue2336-main-rebind-delivery.json` records their commit SHA. They do not
advance the checkout branch or publish it to GitHub.
