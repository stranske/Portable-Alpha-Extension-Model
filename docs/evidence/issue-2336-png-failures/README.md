# Single-run PNG failure evidence for #2336

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
`manifest.json` binds every member's bytes/SHA256 and the current production/test/replay inputs.
`comparison.json` records every new node, incumbent outcome comparison and new covered line.
Earlier evidence directories keep their original execution identities.

Hosted CI, reviews, expected check topology, the seven-minute review floor, guarded merge and
actual post-merge comparison remain the keepalive/closer lifecycle. This evidence does not
claim those stages, provider approval, installed deployment or completion of the broader #2336.
