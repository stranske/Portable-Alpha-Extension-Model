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
| Historical runner | [`run_baseline.py` at `7595bbc`](https://github.com/stranske/Portable-Alpha-Extension-Model/blob/7595bbc0/docs/evidence/issue-2336-png-failures/current-main-20261111/run_baseline.py). The commands below and the retained `run.json` files record the invocations. |
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
- The focused PNG/PDF/sweep suite passed on unmodified production code in attempt 2, with
  named-node JUnit in `provisioned/focused-junit.xml`. Attempt 2 retained no stdout or stderr.
  The retained focused-run logs are from attempt 3 in `dev-extra/`.
- Eight mutation pairs, the failure dispositions and the module increment below 90%: **not
  attempted**. The #2495 triage scoped them out until the baseline above completes.
- No deliberate mutation is present: production hashes are unchanged before and after. The
  existing PDF and next-boundaries evidence directories are untouched by this change.
- Closure eligibility: **false**. The next attempt needs an environment provisioned with
  `-e '.[dev]'` (or with `app-baseline-kit` added to `requirements-dev.txt`), then the same
  full coverage command under the same stop rule.

### Attempt 3 results (raw files: `current-main-20261111/dev-extra/`)

This section supersedes the attempt-2 status and closure statement above, which are kept as
they were recorded.

`dev-extra/run.json` records commit `7595bbc0`, the commit that added the attempt-3
plan. That commit changes only the README relative to its parent. The `pa_core`,
`dashboard` and `tests` trees at `7595bbc0` are byte-identical
to `5be77e15`, and the porcelain for those paths was empty. `pa_core/cli.py` SHA-256 was
`f3ffec1a…40e26` before and after. Versions: pytest 9.2.0, pytest-cov 7.1.0, coverage 7.17.2,
numpy 2.5.4, pandas 3.1.1, openpyxl 3.1.5, kaleido 1.5.0, plotly 6.9.0, SciPy absent.

| Command | Completed / timed out | Exit | Elapsed | Result |
| --- | --- | ---: | ---: | --- |
| full coverage | completed | 1 | 164.1 s | **1 failed, 2116 passed, 5 skipped, 3 deselected**; `dev-extra/coverage.json` written |
| focused PNG/PDF/sweep | completed | 0 | 3.7 s | **23 passed** (`dev-extra/focused-junit.xml`, `focused-stdout.log`) |

The named-node JUnit is `dev-extra/full-junit.xml` (2,122 testcases). stdout/stderr are in
`dev-extra/*-stdout.log` and `*-stderr.log`, and argv, exit, elapsed time and the timeout flag
are in `dev-extra/run.json`.

#### Coverage totals from the committed `dev-extra/coverage.json`

| Scope | Covered | Statements | Percent |
| --- | ---: | ---: | ---: |
| Total (`totals`) | 13,532 | 14,670 | 92.24267211997274% |
| `pa_core/` files | 10,910 | 11,759 | |
| `dashboard/` files | 2,622 | 2,911 | |

The total is 1,138 missing lines and 192 excluded lines. It is at least 90%, so the
below-90% module increment (T6) does not apply and no tests were added.

#### Reconciling the historical claims

- **PR #2475 at head [`d112c223`](https://github.com/stranske/Portable-Alpha-Extension-Model/commit/d112c223):**
  the commit's `current-manifest.json` binds `current-validation.tar.gz` (SHA-256
  `7b8f6122b0e69e310693676459e886b7c6a9e1014c6bc114198fe516a8583e91`) and its
  `final-candidate-coverage.json`, `final-candidate.xml`, and `final-candidate.log` members.
  Rechecking the archive hash and every manifest member confirms **13,537 / 14,670 =
  92.27675528289025%** in that coverage JSON (member SHA-256
  `6be66c3a4a3daa0834a4032f997b841c0ac318d778c8f7bbf4c861a9130ce99f`).
  **13,441 is a transcription error**, not an alternative measurement. Current main's
  13,532 is a separate run, not a correction to that historical numerator. The historical
  console and JUnit identify Python 3.12.2, below the declared >=3.13 floor; this historical
  run does not establish supported-runtime acceptance.
- **83.76% / 12,191 of 14,554** (earlier sections): this measured the older `f9853be5`
  snapshot, which has a smaller statement scope. It is superseded for current main, not
  contradicted.

#### Failure dispositions (one `python -m pytest <node>` reproduction each; logs and JUnit in `dev-extra/repro/`, commands in `dev-extra/repro/commands.txt`)

| Node | Source | Observed now | Supported cause |
| --- | --- | --- | --- |
| `tests/test_schema_export.py::test_parameter_dictionary_matches_schema` | fresh full run and PR #2475 | **fails** on 3.13.13 in the full run and alone (exit 1) | Interpreter-dependent type repr. On 3.13, `typing.Optional[Dict[str, int]]` renders `Optional[Dict[...]]`; on 3.14 (CI) it renders `Dict[...] \| None`, which is what the committed `docs/guides/PARAMETER_DICTIONARY.md` contains (both checked directly). This is not a code-under-test regression, but it is a real portability defect under the declared `requires-python >=3.13` floor. It is not fixed here. |
| `tests/test_cli.py::test_main_with_png` | earlier strengthened run (Kaleido timeout) | **passes**: 8.6 s in the full run, 5.6 s alone (kaleido 1.5.0) | The historical Kaleido timeout does not reproduce. Its raw log and Kaleido version are not in the repository, so the original cause stays **unresolved (record unavailable)**. |
| `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples` | PR #2475 | **passes** (35.0 s full, 32.6 s alone) | #2475 `d112c223` manifest-bound `final-candidate.xml` records `pip wheel ... --no-deps` exiting **1** (`CalledProcessError`). The underlying captured wheel stderr is not exposed in the retained traceback; underlying cause **unresolved**. Not reproduced on current main. |
| `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv` | PR #2475 | **passes** (4.4 s full, 5.3 s alone) | #2475 `d112c223` manifest-bound `final-candidate.xml` records installation exiting **1**; `final-candidate.log` explicitly says `requires a different Python: 3.12.2 not in ">=3.13"`. This is a Python-floor failure, not an unknown entrypoint cause. |
| `tests/test_sleeve_suggestor.py::test_suggest_sleeve_sizes_optimize_prefers_best` | PR #2475 | **skipped** in the venv (SciPy absent), **passes** under bare 3.13.13 with scipy 1.18.1 and numpy 2.5.4 | #2475 `d112c223` manifest-bound `final-candidate.xml` records `assert row["optimizer_success"]` / `assert False`. The optimizer's underlying reason remains **unresolved**. This is distinct from the older NumPy traceback below. |

The schema row above is also directly corroborated by #2475 `d112c223`'s manifest-bound
`final-candidate.xml`: `Parameter dictionary drift detected`, with `Optional[Dict[...]]`
versus `Dict[...] | None`. The schema repair is moved out of this PR.

The older archive currently at `current-validation.tar.gz` (SHA-256
`e5826549174bf763e155ac611077c6272b8ffc032a78a6b23f7c6efa63759a50`, bound by the
current tree's `current-manifest.json`) records a **different** sleeve failure in
`final-candidate.xml`: `AttributeError: module 'numpy' has no attribute 'long'` through
SciPy under `/opt/anaconda3`. It supports an environment incompatibility; it does not
explain #2475's later optimizer assertion. The unverified historical Kaleido timeout has
no retained raw timeout record here and remains separate from both failures. A current
passing reproduction does not resolve an unavailable historical cause.

The four PR #2475 node IDs come from the failure list in the PR body. No PR #2475 node ID is
unavailable. The historical node IDs behind the 1,654 / 1,662 / 1,674-pass snapshots above
are in their archives.

#### Eight PNG mutation pairs on the current `pa_core/cli.py` bytes

`current-main-20261111/replay_current.py` loads the unchanged `replay.py` recipe and rebinds
only its pinned revision to `5be77e15` and its digest to `f3ffec1a…40e26`. It uses the same
eight `CASES` anchors (each occurs exactly once in the current file), the same
`logger.error` to `logger.debug` mutation, the same strict JUnit validator and the same
private `git archive` snapshot. Raw output is in `mutation/pairs/`: per case, the mutant
source, RED/GREEN console and JUnit, plus `results.json` with argv, exits and hashes. The run
took 31 s and exited 0.

| Case | RED exit (AssertionError) | GREEN exit | Mutant SHA-256 | Restored SHA-256 |
| --- | ---: | ---: | --- | --- |
| `renderer` | 1 | 0 | `eb581cf34b53…` | `f3ffec1a90fe…` |
| `module` | 1 | 0 | `3936b6e32967…` | `f3ffec1a90fe…` |
| `io` | 1 | 0 | `1401639d6020…` | `f3ffec1a90fe…` |
| `permission` | 1 | 0 | `1401639d6020…` | `f3ffec1a90fe…` |
| `value` | 1 | 0 | `0da4b4648ead…` | `f3ffec1a90fe…` |
| `type` | 1 | 0 | `0da4b4648ead…` | `f3ffec1a90fe…` |
| `runtime-renderer` | 1 | 0 | `d4a5dd1ec33e…` | `f3ffec1a90fe…` |
| `runtime-other` | 1 | 0 | `d4a5dd1ec33e…` | `f3ffec1a90fe…` |

The eight named executions used **five distinct mutant byte sequences**, as recorded in
`mutation/pairs/results.json` and confirmed against the retained mutant sources.
Every RED console contains `AssertionError`, and every GREEN run was hashed at
`f3ffec1a…40e26`. The CLI test input was `fef801ac24c7…`.

The wrapper also checked restoration on the failure and timeout paths
(`mutation/restoration-paths.json`). It injected a `subprocess.TimeoutExpired` on the first RED
pytest call, and separately a `RuntimeError`, both while the first mutant was in place. In each
case the private snapshot's `pa_core/cli.py` was hashed after `replay.py`'s `finally` had run and
before deletion: `f3ffec1a…40e26` both times. The worktree's `pa_core/cli.py` was
`f3ffec1a…40e26` before and after. The timeout was injected, not waited out.

#### Production and existing evidence compared with the baseline

`git diff origin/main` (`5be77e15`) is empty for `pa_core/`, `dashboard/`,
`docs/evidence/issue-2336-pdf-failures/`, `docs/evidence/issue-2336-next-boundaries/`, and this
directory's `replay.py`, `*.json` and `*.tar.gz`. The worktree's `pa_core/cli.py` is
`f3ffec1a…40e26`. All mutations ran in private `git archive` snapshots, so no deliberate
mutation remains.

#### Scope of PR #2500

The records are historical evidence bound to `5be77e15` (the production inputs and
mutation replay) and `7595bbc` (the dev-extra coverage and focused run). They do not
measure the PR head. The full run completed with 92.24% statement coverage and exit 1
on the parameter-dictionary test. The focused run passed all 23 cases.

#2495 remains open for the schema follow-up. It covers structural rendering of Optional
and Union annotations, regression cases for `Optional[Dict[...]]` and the equivalent PEP 604
union, and dictionary regeneration if the canonical output changes.
