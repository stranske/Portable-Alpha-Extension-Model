# Single-run PNG failure evidence for #2336

## Reproducible bounded capture — 2026-11-04

The [automated capture](automated-capture-20261104/README.md) completes with
**2,029 passed, 2 failed, 3 skipped, 4 deselected** and **13,493 / 14,626 covered
statements (92.25352112676056%)**. Production matches main `06881dbd`; the exact
candidate test overlay, including the existing native PNG subprocess bound, is
retained. The new capture driver records argv, revisions and input hashes before
starting pytest, isolates coverage data, and rejects timed-out captures. Its seven
tests pass, including a real stalled process; all 47 capture/summary/replay tests
pass. Historical optimizer and wheel-build dispositions still prevent issue closure.

## Current-main PNG replay — 2026-11-04

The [main `06881dbd` capture](main-06881dbd-20261104/README.md) re-runs all eight
assertion RED/restored GREEN pairs against the current main revision, with raw
console logs, named-node JUnit, argv and hashes retained individually. The focused
PNG/PDF/sweep suite passes all 23 tests. The replay driver now resolves an explicit
revision once and rejects a caller CLI that differs from that snapshot; its two
new regressions fail before the change and pass afterward. Historical failure
dispositions and issue closure remain open.

## Installed-wheel failure diagnostics — 2026-11-04

The [install diagnostic repair](install-diagnostics-20261104/README.md) preserves
pip stdout/stderr for installed-wheel failures in both online and offline probes.
Two new cases fail before the repair and pass afterward; focused regressions
report **85 passed, 1 skipped, 1 deselected**. Retained coverage and all eight PNG
pairs were independently revalidated. Historical failure dispositions remain open.

## Verified coverage capture — 2026-11-04

The [fresh completed capture](verified-20261104/README.md) reports **2,018 passed,
2 failed, 3 skipped, 4 deselected** and **13,493 / 14,626 covered statements
(92.25352112676056%)** on unchanged current-main production. The evidence helper
now rejects percentages inconsistent with statement counts; all 26 helper tests
pass. The retained eight PNG pairs still match current-main hashes. Historical
optimizer and wheel-build causes remain unresolved, so #2336 stays open.

## Offline installed-wheel verification — 2026-11-04

The [installed-wheel follow-up](offline-install-20261104/README.md) adds a supported-
Python offline installation test. It verifies that product modules and all three
bundled sample paths come from the installed wheel and parses the sample data.
The focused regressions report **57 passed, 1 skipped, 1 deselected**. Historical
optimizer and wheel-build causes remain unresolved, so #2336 stays open.

## Packaging diagnostic repair — 2026-11-04

The [disposition follow-up](disposition-20261104/README.md) preserves wheel-build
stdout/stderr in pytest failure reports, verifies both failure stages with two
regressions, and records a successful offline wheel build on Python 3.14.10.
The focused suite passes **56 tests, with 1 skipped and 1 deselected**. This does
not establish the historical wheel or optimizer failure causes; those tasks and
the issue closure remain open. The reconciled checklist is retained locally
because the GitHub PR-body update requires unavailable approval.

## Retained recheck — 2026-11-04

The [fresh bounded capture](recheck-20261104/README.md) retains console output,
JUnit and raw coverage JSON: **2,008 passed, 2 failed, 3 skipped, 4 deselected**;
**13,493 / 14,626 covered statements (92.25352112676056%)**. It also identifies
nine ignored console logs absent from the previous commit, recovers PR #2475's
historical console from its archive, and validates all eight archived PNG pairs.
The new manifest validator rejects missing or altered evidence files. The two
current failures are environmental; historical optimizer and wheel verification
remain outstanding, so #2336 is not closed.

## Current main measurement — 2026-11-04

The new [capture and failure dispositions](main-20261104/README.md) measure production
at main `7613ffd97a450b15e550f499b7386ec4e67f2c51`, with production code unchanged.
The completed run reports **2,001 passed, 2 failed, 3 skipped, 4 deselected** and
**13,493 / 14,626 covered statements (92.25352112676056%)**. These numbers come
directly from [coverage.json](main-20261104/coverage.json) and the reconciled
[per-module summary](main-20261104/summary.json).

The two failures are the native Kaleido renderer timeout and a network-blocked
setuptools installation. The PNG test now runs its native CLI invocation in a
30-second subprocess; the initial run with untouched tests hung in that renderer
and is retained separately, not presented as a completed measurement. The completed
run includes 13 new evidence-validator tests. As required for this keepalive run,
the slow wheel installation and three live-provider tests are deselected.

All eight current-main PNG assertion RED/restored GREEN pairs passed, as did all
23 focused PNG/PDF/sweep tests. The replay driver now pins the main revision and CLI
SHA256 `d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`.
The new capture preserves the historical archives below, and all PDF and
next-boundaries evidence, unchanged.

PR #2475's archived JSON actually contains **13,537 / 14,670 (92.27675528289025%)**;
its 13,441 numerator is a prose error of 96 statements. It is a separate capture,
not the source of the new measurement. The new Python 3.14.9 / coverage 7.17.0
capture has 44 fewer covered and total statements in `pa_core/wizard_schema.py`
than that Python 3.12 / coverage 7.16.2 capture. The measured file set is identical;
[module deltas](main-20261104/coverage-tool-comparison.json) preserve this distinction.

## Historical evidence

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
