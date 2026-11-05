# Single-run PNG failure evidence for #2336

This directory retains evidence only. It contains no live helper or test changes.
The **supported-runtime current-main acceptance run is still outstanding.** No run
retained here is a coverage acceptance run on current main with Python >=3.13
(`requires-python`). That run, and the per-failure classification that depends on
it, are tracked as follow-up work on #2477.

## Dated historical coverage snapshots — 2026-11-04

Each row is a raw `coverage.json` from one completed bounded pytest run on
2026-11-04. Each is kept with its command record, which holds argv, interpreter,
package versions and the measured main revision. Its summary and, where the run
recorded one, its completion receipt or exit file are kept with it. All of them are
**historical measurements of the revision named**. None of them measures current
main or a supported Python. Every run exited 1. Pass/fail counts come from each
run's JUnit, which this reduction does not keep. The consoles, JUnit, helper
scripts and regressions remain in this PR's history at `5621ee8d`.

| Snapshot | Main revision | Python | Selection | Passed / failed / skipped | Covered statements |
| --- | --- | --- | --- | --- | --- |
| [main-20261104](main-20261104/coverage.json) ([command](main-20261104/commands.json), [summary](main-20261104/summary.json), [exit](main-20261104/completed-suite.exit)) | `7613ffd9` | 3.14.9 | `not slow and not live_llm` | 2,001 / 2 / 3 | 13,493 / 14,626 (92.25352112676056%) |
| [recheck-20261104](recheck-20261104/coverage.json) ([command](recheck-20261104/full-suite-command.json), [summary](recheck-20261104/summary.json), [exit](recheck-20261104/full-suite.exit)) | `7613ffd9` | 3.14.9 | `not slow and not live_llm` | 2,008 / 2 / 3 | 13,493 / 14,626 (92.25352112676056%) |
| [verified-20261104](verified-20261104/coverage.json) ([command](verified-20261104/full-suite-command.json), [summary](verified-20261104/summary.json), [exit](verified-20261104/full-suite.exit)) | `7613ffd9` | 3.14.10 | `not slow and not live_llm` | 2,018 / 2 / 3 | 13,493 / 14,626 (92.25352112676056%) |
| [main-06881dbd-20261104](main-06881dbd-20261104/coverage.json) ([command](main-06881dbd-20261104/full-suite-command.json), [summary](main-06881dbd-20261104/summary.json), [exit](main-06881dbd-20261104/full-suite.exit)) | `06881dbd` | 3.14.9 | `not slow and not live_llm` | 1,988 / 2 / 3 | 13,493 / 14,626 (92.25352112676056%) |
| [automated-capture-20261104](automated-capture-20261104/coverage.json) ([command](automated-capture-20261104/command.json), [summary](automated-capture-20261104/summary.json), [test overlay](automated-capture-20261104/tests.patch)) | `06881dbd` | 3.14.9 | `not slow and not live_llm` | 2,029 / 2 / 3 | 13,493 / 14,626 (92.25352112676056%) |
| [config-guard-20261104](config-guard-20261104/main-capture/coverage.json) ([command](config-guard-20261104/main-capture/command.json), [summary](config-guard-20261104/main-capture/summary.json), [exit](config-guard-20261104/main-capture/full-suite.exit), [test overlay](config-guard-20261104/main-capture/tests.patch)) | `06881dbd` | 3.14.10 | `not slow and not live_llm` | 1,988 / 2 / 3 | 13,493 / 14,626 (92.25352112676056%) |
| [environment-guard-20261104](environment-guard-20261104/main-capture/coverage.json) ([command](environment-guard-20261104/main-capture/command.json), [summary](environment-guard-20261104/main-capture/summary.json), [test overlay](environment-guard-20261104/main-capture/tests.patch)) | `06881dbd` | 3.14.8 | `not slow and not live_llm` | 1,988 / 2 / 3 | 13,493 / 14,626 (92.25352112676056%) |
| [completion-20261104](completion-20261104/main-capture/coverage.json) ([command](completion-20261104/main-capture/command.json), [completion](completion-20261104/main-capture/completion.json), [summary](completion-20261104/main-capture/summary.json), [test overlay](completion-20261104/main-capture/tests.patch)) | `06881dbd` | 3.14.10 | `not slow and not live_llm` | 1,988 / 2 / 3; 1,993 of 1,993 selected nodes finished | 13,493 / 14,626 (92.25352112676056%) |
| [entrypoints-offline-20261104](entrypoints-offline-20261104/candidate-coverage.json.gz) ([command](entrypoints-offline-20261104/candidate-command.json), [summary](entrypoints-offline-20261104/candidate-summary.json)) | `06881dbd` plus PR test changes | 3.14.9 | `not slow and not live_llm` | 2,040 / 1 / 3 | 13,493 / 14,626 (92.25352112676056%) |
| [default-selection-capture-20261104](default-selection-capture-20261104/coverage.json) ([command](default-selection-capture-20261104/command.json), [completion](default-selection-capture-20261104/completion.json), [summary](default-selection-capture-20261104/summary.json), [test overlay](default-selection-capture-20261104/tests.patch)) | `06881dbd` | 3.12.2 (below `requires-python`) | repository default (`not live_llm`) | 2,048 / 4 / 0; 2,052 of 2,052 selected nodes finished | 13,550 / 14,670 (92.3653715064758%) |

These are the failures as recorded. They are not classified as causes:

- The `not slow and not live_llm` runs failed `tests/test_cli.py::test_main_with_png`
  and `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv`. The
  entrypoints candidate run failed only the first of these.
- The default-selection run failed both installed-wheel sample tests in
  `tests/test_dashboard_sample_data.py`,
  `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` and
  `tests/test_schema_export.py::test_parameter_dictionary_matches_schema`. It ran
  on Python 3.12.2, below the package's `requires-python >=3.13`, so it is not a
  supported-runtime measurement.

The test overlays are the disclosed 30-second subprocess bound on
`test_main_with_png` that was applied while capturing. Some command records refer to
helper scripts (`capture.py`, `png_capture_completion.py`, `summarize.py`) by hash.
Those scripts are not part of this reduction. In the completion run, the bytes they
had at capture time are kept as
[capture-input.txt](completion-20261104/main-capture/capture-input.txt) and
[completion-plugin-input.txt](completion-20261104/main-capture/completion-plugin-input.txt).

## Current-CLI PNG failure-path proof — completion-20261104

[png-proof/](completion-20261104/png-proof/) re-runs all eight
`tests/test_cli_png_failures.py::test_single_run_png_failure_preserves_results`
assertion RED / restored GREEN pairs against main `06881dbd`. It keeps the raw
consoles, JUnit and mutant sources for each case, plus
[results.json](completion-20261104/png-proof/results.json).
[png-audit.json](completion-20261104/png-audit.json) records that all eight cases
had a RED assertion, a GREEN pass and a byte-identical restoration. The focused
PNG/PDF/sweep export receipt, [focused.txt](completion-20261104/focused.txt) and
[focused.xml](completion-20261104/focused.xml), reports **23 passed**. Its argv is in
[commands.json](completion-20261104/commands.json).

The proof's input bindings are `pa_core/cli.py` SHA256
`d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e` and
`tests/test_cli_png_failures.py` SHA256
`fef801ac24c71a8a928f958959861ad372f7663982d338ee7a096da11a061dab`. Both are
byte-identical on main `c3190d76` (2026-11-04), so the proof applies to the current
CLI. This is failure-path coverage. It is not evidence that native PNG export works.
[manifest.json](completion-20261104/manifest.json) binds the bytes and SHA256 of
every retained file in this directory. Entries for files this reduction did not keep
were dropped, and the 52 remaining hashes were re-verified. Its `inputs` block keeps
the historical helper/test hashes from capture time unchanged.

## Supported-stack dispositions — 2026-11-04

[dispositions.xml](dispositions.xml) reports **2 passed, 0 failed, 0 skipped,
0 errors** on Python 3.13.13 with NumPy 2.5.4 and SciPy 1.18.1:

- `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best`
- `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples`

Recorded command:

```sh
/opt/homebrew/bin/python3.13 -m pytest -o addopts='' tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples -q --junitxml=docs/evidence/issue-2336-png-failures/dispositions.xml
```

These are two current passes on the recorded stack and nothing more. They do not
establish causes for earlier optimizer or wheel-build failures whose diagnostics are
unavailable.

## Outstanding (follow-up on #2477)

- A coverage run on the then-current main SHA with a repository-selected Python >=3.13
  and its pinned coverage dependency, using the repository's default marker selection
  and a recorded 600-second timeout. It should keep argv, interpreter, dependency
  versions, completion receipt, console and JUnit.
- Classification of `test_main_with_png` under the reviewed 30-second bound, if
  native rendering prevents completion.
- Supported-stack reruns of the wheel and optimizer tests that keep subprocess stderr
  and optimizer diagnostics.
- A README update from that new `coverage.json` that lists every current failure and
  separates observed causes from historical uncertainty.

## Original #2336 evidence

Eight tests run the real CLI simulation/report/manifest path while substituting only the
optional image exporter and visualization objects. They exercise renderer imports, unrelated
missing modules, filesystem and permission errors, invalid values and types, renderer runtime
failures and unrelated runtime failures. Each checks the exact diagnostic and guidance, one
ERROR log, one exporter call, unchanged saved workbook bytes, no purported PNG file, preserved
configuration/seed and the final manifest warning. This is failure-path coverage, not a native
renderer or Office export witness. Production code is unchanged.

## Selection and complete comparison

At baseline `f9853be5b782d0ddc63d5bd1505cd3e686f87d80`, the last 500 source-touching commits
rank `pa_core/cli.py` first: 21 repair-subject proxies, 133 touches and 269 uncovered statements.
Subjects are proxies, not a measured escaped-defect population. The complete source ranking is
retained in the archive. The previous PDF chunk (#2350) is already merged and is not duplicated.

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| Passing tests | 1654 | 1662 |
| Failing tests | 1 | 1 |
| Skipped / default-deselected | 1 / 2 | 1 / 2 |
| Covered statements | 12166 / 14554 | 12191 / 14554 |
| Total statement coverage | 83.59213961797444% | 83.76391370070084% |
| CLI covered statements | 838 / 1107 | 863 / 1107 |

The complete baseline and candidate each exit 1 because
`tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` hits the
pre-existing local SciPy/NumPy import incompatibility. No existing testcase outcome changed,
no covered line regressed, and statement/exclusion scope is identical. Neither result is a
full-suite PASS or 90% acceptance. The focused export suite reports **23 passed**.

Commands used with `/opt/anaconda3/bin/python3`:

```sh
python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:<phase>-coverage.json --junitxml=<phase>.xml tests/
python -m pytest tests/test_cli_png_failures.py tests/test_cli_pdf_failures.py tests/test_cli_sweep_png_export.py -q --junitxml=pa-focused.xml
python docs/evidence/issue-2336-png-failures/replay.py --output <fresh-directory>
```

## Deliberate production break and byte-identical restoration

`replay.py` archives the pinned Git baseline into a private temporary directory and overlays
the candidate test. For each exact node below it changes that production handler's
`logger.error(...)` to `logger.debug(...)`, losing the required ERROR log and manifest warning.
The named test must produce exactly one assertion failure, then exactly one passing testcase
after restoration. Error/skip/partial JUnit results are rejected. Caller inputs remain
unchanged and an existing output directory is refused before mutation.

| Exact parameter of `tests/test_cli_png_failures.py::test_single_run_png_failure_preserves_results` | RED exit / assertion failures | Restored GREEN exit / failures |
| --- | --- | --- |
| `renderer` | 1 / 1 | 0 / 0 |
| `module` | 1 / 1 | 0 / 0 |
| `io` | 1 / 1 | 0 / 0 |
| `permission` | 1 / 1 | 0 / 0 |
| `value` | 1 / 1 | 0 / 0 |
| `type` | 1 / 1 | 0 / 0 |
| `runtime-renderer` | 1 / 1 | 0 / 0 |
| `runtime-other` | 1 / 1 | 0 / 0 |

These are eight named executions over six distinct handler mutations. They prove the tests
reject loss of production error logging; they do not independently mutate every diagnostic,
workbook or manifest assertion. Every restoration matches original production SHA256
`95f0e38b1e2491ff23fe4f278c2d5c5d29a6293bfbf5ef14b64795a4c954906e`.

`validation.tar.gz` retains 51 raw files: complete baseline/candidate console, JUnit and
coverage JSON; the focused result; all eight mutant source files; 16 exact-node console/JUnit
pairs; per-phase argv, exits and source hashes; command metadata and source ranking.
`manifest.json` binds every member's bytes/SHA256 and the production/test/replay inputs at the original implementation head `276f66c1d95e5895f249cb5c75e84756004a64d6`. Those historical hashes are retained; the review repairs below changed the test and replay inputs.
`comparison.json` records every new node, incumbent outcome comparison and new covered line.
Earlier evidence directories keep their original execution identities.

Hosted CI, reviews, expected check topology, the seven-minute review floor, guarded merge and
actual post-merge comparison remain the keepalive/closer lifecycle. This evidence does not
claim those stages, provider approval, installed deployment or completion of the broader #2336.

## Current review repair and replay

The `d9f63594` review repair added the single-ERROR assertion and strict JUnit validation.
A fresh replay exposed a validator defect: pytest writes `assert 0 == 1` in the XML
`message` and places `AssertionError` in the terminal traceback line. Searching only the
message rejects a real assertion; a substring search also accepts a `ValueError` whose
message merely mentions `AssertionError`. The corrected validator accepts an explicit
assertion type or the terminal pytest traceback type. Two regressions invoke real pytest
in disposable directories: both fail against the old validator and pass after the fix.
No production CLI code changed.

`current-validation.tar.gz` retains the current eight exact-node production RED/restored
GREEN pairs, raw validator regression RED/GREEN logs and JUnit, and the complete final
candidate suite log, JUnit and coverage JSON. `current-manifest.json` binds every member
and the current production, CLI-test, validator-test and replay inputs.
`current-comparison.json` compares every old testcase outcome and covered line against
the original baseline; the original archive and comparison keep their original identity.
The local SciPy/NumPy failure remains an explicit non-PASS full-suite result. Broader
#2336, native renderer/Office exports, hosted acceptance and provider disposition remain
separate from this bounded proof.

Final candidate full suite: **1,674 passed, 1 failed, 1 skipped, 2 deselected**.
Against the original baseline there are 20 new PASS nodes (eight CLI cases and
12 validator cases), zero changed/removed old outcomes, unchanged statement/exclusion
scope and zero covered-line regressions. Production coverage remains
12,191 / 14,554 statements (83.76391370070084%). The current archive has 49 members
and four current input bindings. Focused validator/CLI validation reports **20 passed**.
