# Verify selected tests finish before accepting full-suite evidence

**13,493 / 14,626 covered statements (92.25352112676056%).** These values equal
[main-capture/coverage.json](main-capture/coverage.json) exactly and are printed in
[capture-summary.txt](capture-summary.txt). The completed run reports **1,988 passed,
2 failed, 3 skipped, 4 deselected**, exit 1, in 148.34 pytest seconds (151.33 seconds
including startup and reporting). Every one of the **1,993 selected nodes finishes**;
[main-capture/completion.json](main-capture/completion.json) retains their identities.

[main-capture/summary.json](main-capture/summary.json) records every module's coverage,
missing line numbers, and the 15 largest source gaps. The raw console and named-node
JUnit are [main-capture/full-suite.txt](main-capture/full-suite.txt) and
[main-capture/full-suite.xml](main-capture/full-suite.xml).

## Main revision, inputs and command

The GitHub connector verified current main as
`06881dbdf0e915c305669e7fe3f8baa7d8a21ec4`. A separate local clone checks out that
exact commit. Product files and test/coverage configuration match main. The only
test overlay is the existing 30-second native PNG subprocess bound, preserved in
[main-capture/tests.patch](main-capture/tests.patch). Candidate test additions do
not contribute to this main-suite pass count.

The command, with absolute interpreter and artifact paths in
[main-capture/command.json](main-capture/command.json), is:

```sh
timeout --signal=INT --kill-after=10s 600s python -m pytest -p png_capture_completion --capture-completion-json=completion.json --cov=pa_core --cov=dashboard --cov-report=json:coverage.json --junitxml=full-suite.xml -m 'not slow and not live_llm' tests/
```

The marker expression applies the required slow exclusion and preserves the default
live-provider exclusion: one online wheel test and three provider tests are deselected.
Python is 3.14.10; pytest is 9.2.0, pytest-cov 7.1.0, and installed coverage 7.17.0.
Main pins coverage 7.17.1; this records the actual installed toolchain.

The completion plugin is loaded through the recorded `PYTHONPATH`. The measured
driver and plugin bytes are preserved in `main-capture/capture-input.txt` and
`main-capture/completion-plugin-input.txt`; their hashes match the command metadata.
The retained plugin input precedes Black's whitespace-only formatting of its file.

| Largest source gap | Missing statements |
| --- | ---: |
| `pa_core/cli.py` | 165 |
| `pa_core/sleeve_suggestor.py` | 115 |
| `dashboard/pages/3_Scenario_Wizard.py` | 101 |
| `dashboard/pages/5_Scenario_Grid.py` | 57 |
| `pa_core/sim/paths.py` | 42 |

## Completion defect and regression proof

Exit 1 and internally consistent JUnit/coverage do not establish that all selected
tests ran. A runtime scheduling change can leave selected nodes unexecuted while
pytest still writes both reports. The new real-pytest regression selects three
nodes, truncates the session's remaining items in the first test, and fails that
test. The old helper incorrectly reports the resulting one-test run as completed;
[driver-red.txt](driver-red.txt) records the assertion failure against that helper.

The plugin snapshots selected node IDs after collection and records each completed
runtest protocol, including setup/teardown handling. The driver compares node-ID
multisets and subprocess exit codes before accepting evidence. A missing receipt
or unfinished node leaves `completed=false` and produces no summary. It also
records the deselection count independently of console parsing.

The paired complete run contains one failing and two passing nodes and remains
valid completed evidence. Both real-pytest cases and all existing helper tests pass:
**62 passed** in [regressions.txt](regressions.txt) and [regressions.xml](regressions.xml).
The whole-tree Black check passes for **537 files**; Ruff passes. A serial Black
check populated a writable cache before the required command, because this sandbox
restricts multiprocessing IPC. This change affects evidence capture, not product code.

## Fresh export and PNG mutation proof

[focused.txt](focused.txt) and [focused.xml](focused.xml) retain **23 passing**
PNG/PDF/sweep tests. [commands.json](commands.json) records focused and replay argv.
All eight fresh PNG pairs have assertion RED and restored GREEN results at current
main's CLI SHA256:

`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`

[png-proof/results.json](png-proof/results.json) retains each exact node, argv,
exit, mutation anchor and original/mutant/restored hash. Individual console and
named-node JUnit files are directly readable in `png-proof/`. Mutant source bytes
are gzip-compressed. [png-audit.json](png-audit.json) records an independent check
of all eight XML pairs, mutant hashes, and byte-identical restoration to main.
The caller's production files remain unchanged. PDF and next-boundaries evidence
remain untouched.

## Current and historical failure dispositions

| Test | Cause or remaining uncertainty |
| --- | --- |
| `tests/test_cli.py::test_main_with_png` | This completed run exceeds the 30-second native renderer subprocess bound. The retained [independent browser probe](../main-06881dbd-20261104/browser-probe.txt) establishes sandbox Chromium startup restrictions (`setsockopt: Operation not permitted`). Environmental; native PNG success is not established. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | This main run cannot resolve PyPI while installing setuptools, before entrypoint behavior. Environmental. PR #2475 instead used Python 3.12.2, incompatible with the package's Python >=3.13 floor. |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | Historical unsupported Python 3.12 rendered schema type text differently. The unchanged test passes in this supported-Python main run. Environmental. |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | Historical build stderr was discarded, so the exact cause remains unresolved. This slow online test is deselected as required. Prior supported-Python offline wheel verification does not recover the historical cause. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | The historical assertion and SciPy warning do not distinguish import incompatibility from solver nonconvergence. This run skips because SciPy is absent; compatible-SciPy execution remains unavailable. Historical cause unresolved. |

The other two skips require absent `langchain_anthropic`. Historical PR #2475's
JSON is **13,537 / 14,670 (92.27675528289025%)**; 13,441 was a prose error of 96
statements. The earlier [reconciliation](../main-20261104/pr2475-reconciliation.json)
and [tool comparison](../main-20261104/coverage-tool-comparison.json) remain unchanged.
Their toolchains differ by 44 covered and total statements in `pa_core/wizard_schema.py`.

## Verified task state

- [x] Completed bounded current-main coverage, including counts, modules and source gaps.
- [ ] Conclusively disposition all four historical failures and the PNG timeout.
- [x] Fresh 23-test focused run and eight assertion RED/restored GREEN PNG pairs.
- [ ] Publish final evidence and close #2336 after historical dispositions are conclusive.
- [x] Retain completed raw coverage JSON, printed percentage, command and main SHA.
- [x] README covered/total statements equal coverage JSON exactly.
- [x] All restored PNG production hashes equal current main's CLI hash.
- [x] Every failure in this completed main run has a cause recorded above.
- [x] Total product coverage exceeds 90%.

These statuses apply to the duplicate task lists: **14/18 verified**, with historical
disposition and conditional closure incomplete. GitHub reports #2336 already closed
with reason `completed`; this run did not change that state or establish the two
missing historical causes. PR #2478 was verified open and ready for review.

The checkout's `.git` is read-only. A delivery commit, patch and bundle are created
in a writable clone under `/tmp/issue2336-completion-delivery`; this does not advance
the checkout branch or publish its commit. `manifest.json` binds the retained
evidence artifacts and source/test inputs.

The GitHub PR checklist update was rejected with `MCP tool call requires approval,
but approval policy is never`. The reconciled body, checking only the 14 verified
items and preserving the four incomplete ones, is saved in
`/tmp/issue2336-completion-pr-body.md`. No remote checklist update was made.
