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

## Current-main baseline for #2495 (recorded before execution)

Narrowed scope from the #2495 park triage: one bounded current-main baseline, the full
coverage run and the focused PNG/PDF run, then report completed versus timed-out runs and
the first concrete failure. Mutation pairs and coverage improvements are out of this pass.

| Field | Value |
| --- | --- |
| Check date | 2026-11-11 |
| Tested main commit | `5be77e15c925f5354ab67e2a1749fe6187d2038c` (`origin/main` at start) |
| Working tree before execution | clean (`git status --porcelain` empty) |
| `pa_core/cli.py` SHA-256 | `f3ffec1a90fe5888ca5ba357507e7542f4dc436117acea430348c87ab0b40e26` |
| Interpreter | `/opt/homebrew/bin/python3.13` (3.13.13). `pyproject.toml` declares `requires-python = ">=3.13"`; CI uses 3.14, which is not installed locally. Earlier evidence in this directory ran on 3.12, below the declared floor. |
| Runner | `current-main-20261111/run_baseline.py` (records argv, exit, elapsed, timeout flag, interpreter and dependency versions, production hashes) |
| Operational outcome | Both commands reach a terminal exit inside the stop rule with artifacts committed; the focused run exits 0. |
| Maximum attempts | 1 per command (no reruns to obtain a green result) |
| Timeout / elapsed-time stop rule | 600 s per command; on expiry the process is killed, the run is recorded as `timeout: true` and is not a completed run. The last measured full run took 182 s. |

Commands (run from the repository root, serial because the issue names the exact command):

```sh
python -m pytest --cov=pa_core --cov=dashboard --cov-report=json:coverage.json tests/ --junitxml=<evidence>/full-junit.xml
python -m pytest tests/test_cli_png_failures.py tests/test_cli_pdf_failures.py tests/test_cli_sweep_png_export.py -q --junitxml=<evidence>/focused-junit.xml
```

### Attempt 2 declared after attempt 1 stopped on the environment (recorded before execution)

Attempt 1 (raw files directly in `current-main-20261111/`) completed inside the stop rule but
measured the interpreter, not the code: bare `/opt/homebrew/bin/python3.13` has no `openpyxl`,
so the full run stopped with 23 collection errors and the focused run failed 23 of 23. This
second attempt changes only the environment, not the commands, code or tests: a fresh venv
`/tmp/pae-2495-venv` from the same 3.13.13 interpreter, provisioned with
`uv pip install -r requirements.txt -r requirements-dev.txt -e .`. SciPy is not declared by the
repository and is not installed; `pa_core/sleeve_suggestor.py` treats it as optional.
Same commands, same 600 s timeout, one attempt, output in `current-main-20261111/provisioned/`.

### Attempt 3 declared in keepalive round 2 (recorded before execution)

Attempt 2 stopped on the declaration gap described under Results: `requirements-dev.txt` does not
install `baseline_kit`. Attempt 3 changes only the environment again. It uses the same
`/tmp/pae-2495-venv` (3.13.13), which `uv pip install -e '.[dev]'` has also provisioned,
adding `app-baseline-kit==0.2.0` from `stranske/Workflows@49a88b3776e7cc60d13c670d52c81b8303dfec0a`
(`packages/app-baseline-kit`), plus `pytest-regressions` 2.11.0 and `pytest-datadir` 1.9.0. It
runs the same two commands, the same `run_baseline.py` and the same 600 s timeout, with one
attempt. `origin/main` is still `5be77e15`, and `git diff 5be77e15 HEAD -- pa_core dashboard tests
pyproject.toml requirements*.txt` is empty. Output goes to `current-main-20261111/dev-extra/`.
The stdout/stderr logs are force-added because `*.log` is gitignored, which is why attempts 1 and
2 committed no logs. Operational outcome for this attempt: the full run finishes without a
collection interruption and writes `dev-extra/coverage.json`. If it does not, the first failure
is recorded and no fourth attempt is made in this round.

### Results (raw files: `current-main-20261111/run.json` and `current-main-20261111/provisioned/run.json`)

Production code under test is byte-identical to `5be77e15` in both attempts. Attempt 2's
`run.json` records `4d04cef0`, the commit that added this plan; it changes only this directory.
`pa_core/cli.py` SHA-256 was `f3ffec1a…40e26` before and after both attempts; no production
file was modified, and `git status --porcelain -- pa_core dashboard tests` was empty.

| Attempt | Command | Completed / timed out | Exit | Elapsed | Result |
| --- | --- | --- | ---: | ---: | --- |
| 1 (bare 3.13.13) | full coverage | completed | 2 | 7.1 s | collection interrupted: 23 errors, `ModuleNotFoundError: No module named 'openpyxl'`; no `coverage.json` written |
| 1 (bare 3.13.13) | focused PNG/PDF/sweep | completed | 1 | 2.7 s | 23 failed, all `ModuleNotFoundError: No module named 'openpyxl'` |
| 2 (provisioned venv) | full coverage | completed | 2 | 22.5 s | collection interrupted: 4 errors in `tests/baseline/`, `ModuleNotFoundError: No module named 'baseline_kit'`; no `coverage.json` written |
| 2 (provisioned venv) | focused PNG/PDF/sweep | completed | 0 | 3.3 s | **23 passed** (named-node JUnit in `provisioned/focused-junit.xml`) |

Attempt 2 dependency versions: pytest 9.2.0, pytest-cov 7.1.0, coverage 7.17.2, numpy 2.5.4,
pandas 3.1.1, openpyxl 3.1.5, kaleido 1.5.0, plotly 6.9.0, SciPy absent.

**First concrete failure of the full run:** `requirements-dev.txt` does not install the
`app-baseline-kit` package (module `baseline_kit`) that `tests/baseline/*.py` import. It is
declared only in the `dev` extra in `pyproject.toml`, as an unpinned
`git+https://github.com/stranske/Workflows.git` reference. `pr-00-gate.yml` installs
`requirements-dev.txt`. Whether hosted CI collects `tests/baseline/` some other way is not
established here. This is an environment and declaration gap, not a code-under-test failure.
The stop rule (one attempt per declared environment) tripped here, so no third run was made.

Status against #2495 acceptance:

- Completed full coverage run with committed `coverage.json`: **not met**. Both attempts
  completed inside the stop rule, but both were interrupted at collection, so no coverage
  total exists for current main. The historical 92%, 13,441 and 13,537 claims remain
  unreconciled, and the 83.76% / 12,191 of 14,554 figure above remains the last measured total.
- Focused PNG/PDF/sweep suite passes against unmodified production code with logs and
  named-node JUnit: **met** (attempt 2).
- Eight mutation pairs, the failure dispositions and the module increment below 90%: **not
  attempted**. The #2495 triage scoped them out until the baseline above completes.
- No deliberate mutation is present: production hashes are unchanged before and after. The
  existing PDF and next-boundaries evidence directories are untouched by this change.
- Closure eligibility: **false**. The next attempt needs an environment provisioned with
  `-e '.[dev]'` (or with `app-baseline-kit` added to `requirements-dev.txt`), then the same
  full coverage command under the same stop rule.
