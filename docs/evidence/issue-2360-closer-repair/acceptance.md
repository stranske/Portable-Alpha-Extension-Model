# Issue #2360 acceptance recheck

## Coverage summary binding contract (2026-11-10 UTC)

Before source/test edits, inspected both historical `pyproject.toml` revisions
with `git show`, their eight-case-only diff, retained paired-CI receipts and
live run 42887512783's successful parent/merge jobs. Baseline: 95 evidence tests
pass; the acceptance command prints six PASS results and OVERALL PASS. The
remaining unchecked coverage entries have retained measurements, but the
verifier does not bind the comparison's `target_units` counts to raw reports.

Expected outcome from `python scripts/verify_issue_2360_evidence.py`: print
statement/branch denominators, uncovered counts before/after and measured
reduction; contradictory recorded counts must produce coverage FAIL and
OVERALL FAIL (exit 1). Add focused regression tests rather than repeating the
historical suite. Stop after two retrieval attempts per unavailable receipt or
45 minutes total. `gh issue view 2360` and `gh pr view 2372` exited 1 because
shell network access is unavailable; the connector re-read #2360 and confirmed
PR #2485 is open with `draft=false` at `efc96609`. Prior CI, mutation and ranking
findings remain separate, including explicit ranking provenance UNKNOWN.

Verified result: all eight new corrupted-summary cases failed before the fix.
After the fix, **120 targeted tests pass**, with **98%** verifier statement
coverage. The recorded acceptance command prints six PASS results and
OVERALL PASS; its coverage line now reports **463 to 390 uncovered units**, with
**1050 statements + 284 branches**, and **73 fewer units**. Ruff and
`git diff --check` pass. Both newly inspected isolated checkouts have the
recorded identical `pyproject.toml` hash `b442a73d…`; only the original eight-case
test file differs. Existing paired CI provides execution evidence; no local
full-suite rerun or later test contribution is claimed.

Black formatted both changed Python files individually. Its required repository
check passed with **535 files unchanged**, using one worker and the temporary
standard asyncio wake-up timer already described in this record. Default
multiprocessing was rejected by the sandbox; an unnecessary subsequent diagnostic
retry with a shorter 20-second bound timed out. The completed full check remains
the formatting evidence. Commands and results are in `coverage_summary_binding_followup`
in [validation.json](./recheck/validation.json); actual acceptance output is in
[verification-output.txt](./recheck/verification-output.txt).

- [x] Verify comparable configured-suite statement-plus-branch attribution at
  adjacent parent/original merge, including raw denominators and uncovered
  counts, without crediting later tests.
- [x] Verify the acceptance coverage criterion: the retained configured-suite
  command ran successfully on both revisions and measured a 73-unit reduction.

Staging exited 128 because this checkout's `.git/index.lock` is read-only.
Prepared the verified source commit in `/tmp/issue2360-count-binding` and an
importable patch at `/tmp/issue2360-count-binding.patch`; the checkout's branch
cannot be advanced here. This delivery constraint does not change attribution.

After validation, attempted to reconcile only the two current unchecked PR
coverage entries and replace the obsolete top-level UNKNOWN outcome, preserving
the nested historical issue. The GitHub tool rejected publication with
`MCP tool call requires approval, but approval policy is never`. Remote
checkboxes remain unchanged. PR #2485 was re-read open with `draft=false` at
`efc96609` immediately before that attempt. The source commit and patch remain
available locally; no branch update or PR-body publication is claimed.

Check date: **2026-11-07 UTC**. This contract was recorded before implementation
or evidence changes for issue #2474. Historical receipts below this directory
retain their original dates; they are not new measurements.

Expected output: one `PASS`, `FAIL`, or `UNKNOWN` for each requirement (revision
binding, exact-head executed pytest CI, eight original mutation cases, isolated
coverage attribution, ranking provenance, record completeness), followed by
`OVERALL PASS` only if every mandatory requirement passes. Missing mandatory
CI or per-case mutation evidence must produce `OVERALL FAIL` and exit 1.

Runnable verification command (to be implemented in this round):

```sh
python scripts/verify_issue_2360_evidence.py
```

Finite stop rule: at most **two attempts per unavailable remote receipt or
historical execution**, and **45 minutes total** for this round. A missing
receipt stays UNKNOWN; failed tests stay FAIL. Stop retries at that boundary,
record the failing command and limitation, and do not mark blocked tasks done.

## Follow-up reconciliation contract (2026-11-07 UTC)

Reviewed commit `5946ea205281269d8dcda53c73d1293b79e0df22` before
changing source. Its receipts already implement the six distinct tasks above;
the repeated unchecked tasks in PR #2485 are stale. The recorded verification
command exits 0 and all 17 existing evidence tests pass. Reconciliation applies
to the 22 current task/acceptance checkboxes, preserving the quoted original
issue as historical text. No full-suite or ranking compliance is inferred.

The follow-up uses the same verification command, required PASS/FAIL/UNKNOWN
output, and finite stop rule (two remote attempts per receipt, 45 minutes).
The expected outcome is PASS for inspected receipts and FAIL if a successful
job is bound to another SHA, run, attempt, or repository. Before implementation,
the connector re-read run 41661022529 and its jobs: completed success at the
assessed SHA, with the pytest job and step successful. `gh run list` and
`gh run view` again exited 1 because shell network access is unavailable.

Updating PR #2485's body was rejected with `MCP tool call requires approval,
but approval policy is never`. The verified checkbox reconciliation cannot be
published from this environment. This local record retains the completed work;
the remote checkbox update remains a delivery blocker.

## Revision binding follow-up (2026-11-07 UTC)

Before code or test changes, re-read #2360 and #2372/#2467 using the GitHub
connector, including exact base/head/merge SHAs and changed files. The requested
`gh issue view 2360`, `gh pr view 2372`, and `gh pr view 2467` reads exited 1
because shell network access remains unavailable. The connector metadata agrees
with [revisions.json](./recheck/revisions.json). PR #2485 remains open and
`draft=false` at `692a0b6877a6c536ed97c9d55e3bcc8fca3e4502`.

The first repeated unchecked task has retained evidence, but its validator only
checks some recorded fields. This follow-up will reject altered issue identity,
PR base/head/merge SHAs, merged state, source URLs and changed-file lists, and
duplicate PR entries. The adjacent squash parent used for coverage remains
distinct from the API base of #2372. Baseline: all 36 evidence tests pass and
the acceptance command prints six PASS results.

Expected output: `python scripts/verify_issue_2360_evidence.py` prints one
PASS/FAIL/UNKNOWN per requirement and OVERALL PASS only for valid receipts;
altered revision evidence must print FAIL and OVERALL FAIL with exit 1.
Finite stop rule: two attempts per unavailable remote receipt, 45 minutes total.
No historical measurement is repeated unless retained evidence fails inspection.
Remote checkbox publication remains blocked by the previously recorded approval
policy rejection; no checkbox is newly marked before validation.

- [x] Re-read requirements, merged state, exact base/head/merge revisions and
  changed files for the first repeated unchecked task, and record this dated
  contract before code/test changes. Verified against live connector reads.

Twelve new regression cases reproduced incorrect acceptance of altered receipts
before the fix (12 failed, 6 passed). After the fix, all **71** targeted evidence
and wizard tests pass; the recorded acceptance command prints six PASS results
and OVERALL PASS, and Ruff passes. Black formatted each changed Python file and
its full repository check passed with **535 files unchanged**, using the same
temporary single-worker/event-loop timer accommodation documented above.
See [validation.json](./recheck/validation.json), `revision_binding_followup`,
for commands and exits. No coverage measurement or ranking claim was changed.
Remote task checkboxes remain unchanged because publication is blocked.

Staging this follow-up exited 128: `.git/index.lock` is read-only. The verified
source and test changes remain in the working checkout; a writable disposable
clone under `/tmp` holds the source commit, with an importable patch at
`/tmp/issue2360-revision-binding.patch`. This does not advance the checkout's
branch or publish changes to PR #2485.

## Initial inspection

### CI terminal-result follow-up (2026-11-07 UTC)

Before source or test edits, reviewed commits `5946ea20`, `c62ed299`,
`692a0b68` and `8897f8f7`. The retained verifier prints six PASS results and
OVERALL PASS (exit 0); all 54 evidence tests pass. Live connector reads confirm
the original issue requirements, both merged PR revisions and successful run
`41661022529` at the assessed head. Requested `gh` reads still exit 1 because
shell network access is unavailable. PR #2485 is open, `draft=false`, at
`8897f8f7dbba3fd3a4bae6dde43414be72b639d1`.

The six unchecked current task/acceptance entries duplicate verified work.
Attempting to reconcile those entries while preserving the nested original
issue was rejected: `MCP tool call requires approval, but approval policy is
never`. Remote checkbox publication remains blocked; no update is claimed.

Continue with the next unchecked task's CI validation: its log check currently
accepts zero passing tests or a summary containing both passing and failed tests.
Add regression cases and require a successful terminal pytest summary consistent
with the successful job receipt. Expected output from
`python scripts/verify_issue_2360_evidence.py` remains PASS/FAIL/UNKNOWN per
requirement and OVERALL FAIL (exit 1) for contradictory or missing CI results.
Stop after two attempts per unavailable receipt or 45 minutes total. Retained
historical measurements will not be repeated unless inspection invalidates them.

Verified: all six new regression cases failed before the fix; all **77** targeted
tests pass afterward, with 97% verifier statement coverage. The acceptance command
prints six PASS results and OVERALL PASS; Ruff and the full repository Black check
pass (535 files unchanged, using the temporary runtime accommodation above).
Live job `126004180807` and its pytest step both report completed success at the
assessed SHA. See `ci_terminal_result_followup` in
[validation.json](./recheck/validation.json). No historical coverage or ranking
finding changed. Staging exited 128 because `.git/index.lock` is read-only;
a source commit and patch are prepared in a writable disposable clone under
`/tmp/issue2360-ci-delivery`, without advancing the working branch.

### Coverage validation follow-up (2026-11-07 UTC)

Before source edits, reviewed `5946ea20` and `c62ed299`, reran the recorded
verification command (exit 0) and all 27 evidence tests (pass), and re-read
#2360, #2372, #2467 and #2485 through the connector. The original eight cases,
exact-head CI, measured wizard-selected coverage and explicit ranking UNKNOWN
disclosure are already verified. All 22 current task/acceptance checkboxes can
be reconciled from these receipts; the nested original issue remains historical.
The PR body update was attempted and rejected with `MCP tool call requires
approval, but approval policy is never`; its remote checkboxes remain stale.
PR #2485 is open and `draft=false` at `c62ed299`.

This follow-up strengthens coverage attribution validation: require real branch
data, check counts against the recorded statement/branch identities, and reject
changed coverage boundaries even when their sizes match. Expected output remains
requirement-by-requirement PASS/FAIL/UNKNOWN from
`python scripts/verify_issue_2360_evidence.py`, with OVERALL FAIL for invalid
coverage evidence. The stop rule remains two attempts per unavailable receipt
or historical execution, 45 minutes total. No historical measurement will be
rerun unless the retained receipts fail inspection.

`gh run list/view` again exited 1 because shell network access is unavailable.
The connector independently read run `41661022529` and its jobs: completed
success at `a2f4ef50403a0a90e8535e4daef5e6c691037bd2`, with job `126004180807`
and its pytest step successful. Existing immutable receipts remain unchanged.

Verified result: nine new corrupted-receipt cases failed before the fix. After
the fix, all 53 targeted tests pass, verifier statement coverage is 97%, Ruff
passes and the required repository Black check reports 535 files unchanged
(one worker and the temporary asyncio timer described in `validation.json`).
The acceptance command still exits 0 with six PASS results. Raw statement and
branch identities confirm the retained 1050 + 284 boundary and 463 to 390
uncovered units: **73 fewer units in the wizard selection only**. Missing or
inconsistent raw branch data now prevents acceptance. No later tests contribute
to that historical measurement. See
[validation.json](./recheck/validation.json), `coverage_validation_followup`.

Staging this follow-up again failed (exit 128): `.git/index.lock` is read-only.
The working files remain available for automation to commit; a source commit
and importable patch will also be prepared in a writable disposable clone.

The requested `gh issue view 2360`, `gh pr view 2372`, and `gh pr view 2467`
commands were attempted with JSON fields on 2026-11-07. Each exited 1 with
`error connecting to api.github.com`. Read-only GitHub connector requests then
recovered the issue body and both PRs (including changed filenames).

- [Issue #2360](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2360):
  closed; asks for about 60 fewer uncovered statement-plus-branch units, real
  break/revert proofs for each new test, and passing pytest in CI. Its alternative
  requires explaining each target that could not meaningfully be covered.
- [PR #2372](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2372):
  merged; API base `e7c898dea47be8871dac1a29610be6679cdcbd57`, head
  `9bfb197c8ebb6f84f48b9beeac4e8cf6263d9847`, merge
  `e8967edb2aa1bdbc1a68ea58ed241cf205915d03`. Only changed file:
  `tests/test_wizard_render_coverage.py` (eight parameterized cases).
- The locally inspectable merge is a squash with parent
  `10ac6839518576d4dc5199a24392eadfc388c713`. Use this adjacent parent for isolated
  attribution; distinguish it from the API base above.
- [PR #2467](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2467):
  merged; base `6abacb50a941ce1286e74afb5782d3c804427418`, assessed head
  `a2f4ef50403a0a90e8535e4daef5e6c691037bd2`, merge
  `bb8dd930e085b5c3bbc02b30b9f890eac537328c`. Only changed files:
  `docs/evidence/issue-2360-closer-repair/README.md` and
  `docs/evidence/issue-2360-closer-repair/ranking-provenance.json`.

The head objects are absent locally; both squash merges and the #2372 parent
are available. PR #2467's body cites final-head run `41661022529`; that claim
still needs inspection of the actual run/jobs. Existing precursor run
`41660173230` tested a different head and cannot establish final-head acceptance.

## Evidence boundaries

- [mutation-transcripts.md](./mutation-transcripts.md) and
  [mutations.json](./mutations.json): historical tails for 17 cases; original
  eight must be distinguished from nine cases added by #2411.
- [README.md](./README.md): historical cumulative 321-unit gain over
  `10ac6839..cc704259`, not measured #2372 gain. The 54-line crosswalk in #2372
  is a projection.
- [ranking-provenance.json](./ranking-provenance.json): original dispatch receipt
  and incident mapping are UNKNOWN. An older #2336 archive is not causal proof.

## Verified recheck results

- [x] Read original requirements and PR metadata; record this contract before
  implementation. [revisions.json](./recheck/revisions.json) retains exact API
  base/head/merge revisions, changed files, and the connector fallback limitation.
- [x] Inspect completed CI for #2467's exact assessed head, including an executed
  successful pytest step. No dispatch or workflow edit was necessary.
- [x] Reproduce all eight original named deliberate-break failures and restored
  passes at #2372's merge. Keep nine later cases separate.
- [x] Measure comparable statement-plus-branch coverage at the adjacent parent
  and merge using identical wizard selection and repository configuration. Keep
  full-suite attribution UNKNOWN and historical cumulative gains separate.
- [x] Search original issue/PR/workflow/local evidence for ranking provenance;
  explicitly disclose the unrecoverable dispatch receipt and incident mapping.
- [x] Implement and run the acceptance verifier, including tests that prevent a
  PASS with missing CI or per-case mutation proof.

These are the six distinct work items; the source issue repeats them under
additional tasks. The details and limitations below apply to every repetition.

### CI: executed pytest, exact assessed head

The GitHub connector fetched the actual
[CI run 41661022529](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41661022529),
not just the PR's assertion. [ci-run.json](./recheck/ci-run.json) records completed
`success`, `workflow_dispatch`, workflow `.github/workflows/ci.yml`, run attempt 1,
and tested head `a2f4ef50403a0a90e8535e4daef5e6c691037bd2`.

[Pytest job 126004180807](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41661022529/job/126004180807),
`Python CI / python 3.14`, completed successfully. Its step
`Pytest (unit tests with coverage)` also completed with `success`.
[ci-jobs.json](./recheck/ci-jobs.json) preserves job/step results;
[ci-pytest-job.json](./recheck/ci-pytest-job.json) preserves selected raw REST
job fields and the pytest step, including the job's own head SHA, run attempt,
run URL and job URL. Unlike the normalized connector list, these fields bind
the successful job independently to the assessed run and repository. The
follow-up verifier rejects a different SHA, attempt, run, repository, unlisted
job or unsuccessful pytest step, and fails overall if this receipt is missing.
[ci-pytest-excerpt.txt](./recheck/ci-pytest-excerpt.txt) preserves selected actual
log lines showing checkout SHA, argument construction, invocation, and result:

```text
1969 passed, 1 skipped, 994 warnings in 132.84s
Coverage 91.25% meets minimum 60.00%
```

The command (matching argument construction, configured inputs, and PR receipt):

```sh
pytest -p ci_test_growth --junitxml=pytest-junit.xml --cov \
  --cov-config=pyproject.toml --cov-report=xml:coverage.xml \
  --cov-report=term-missing --cov-report=json:coverage.json \
  -n auto --dist=loadgroup
```

For independent retrieval when shell network access is available:

```sh
gh run list --repo stranske/Portable-Alpha-Extension-Model \
  --commit a2f4ef50403a0a90e8535e4daef5e6c691037bd2 \
  --json databaseId,workflowName,headSha,status,conclusion,url
gh run view 41661022529 --repo stranske/Portable-Alpha-Extension-Model \
  --json headSha,status,conclusion,url,jobs
gh run view 41661022529 --repo stranske/Portable-Alpha-Extension-Model \
  --job 126004180807 --log
```

These `gh run list/view` commands were attempted here and exited 1 (network
unavailable); equivalent run/job/log reads succeeded through the connector.
The commit-run connector listing omitted this dispatched run; fetching its
immutable run API URL directly recovered it. This omission is not evidence of
an absent CI run.

[Gate 41661029334](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41661029334)
passed but its `Python CI` job was **skipped**; see
[gate-jobs.json](./recheck/gate-jobs.json). It is not the pytest receipt.
The earlier `41660173230` run remains historical, on a different SHA.

### Original mutations: measured reproduction

[mutation-plan.json](./mutation-plan.json) names the exact original eight case
identifiers and literal before/after edits. The
[replay script](../../../scripts/replay_issue_2360_mutations.py) creates a
disposable shared clone, checks out only the original merge, and restores the
page's original bytes in `finally` after every edit. No production file in the
working checkout was mutated.

```sh
python scripts/replay_issue_2360_mutations.py --output /tmp/issue2360-mutation-replay
```

Actual durable output is under [recheck/mutations](./recheck/mutations/):
[mutation-replay.json](./recheck/mutations/mutation-replay.json) contains all
commands, names, exits (eight red=1, eight green=0), output locations and hashes.
All red outputs name their intended case with `1 failed`; every restored output
has `1 passed`. Console receipts remove trailing whitespace only, as declared
in the replay record. Restored original merge page SHA-256:
`6a768b6a8bc41089b32cbf7eb32cd3d005e3db1934d5791cf3daee2710cea654`.

[mutation-transcripts.md](./mutation-transcripts.md) lists the eight original
and nine later cases in separate tables. [mutations.json](./mutations.json)
retains historical receipt tails for all 17; those tails and their later page
hash are not substituted for the new original-merge reproduction.

### Coverage attribution: 73 units in an identical wizard selection

Inspected `git show e8967edb:pyproject.toml` and the local CI consumer. Both
assessed revisions configure pytest `tests/`, default `not live_llm`, coverage
sources `pa_core`, `dashboard`, `design-system`, and `branch=false`. The current
CI consumer delegates pytest to Workflows; the exact historical CI command is
recorded above. `git diff 10ac6839 e8967edb` confirms only the eight-case test file
was added; production code and coverage configuration are identical.

Isolated checkouts: a disposable clone at the squash merge and a worktree of
that clone at its adjacent parent. Creating worktrees from the main checkout
failed because its `.git` directory is read-only; the disposable clone's own
metadata is writable. No later tests or current production files were copied in.

Run this same command **inside each checkout**, using a different JSON output
location per revision:

```sh
python -m pytest tests/ -q -k wizard -m 'not slow and not live_llm' \
  --cov --cov-config=pyproject.toml --cov-branch \
  --cov-report=json:coverage.json --cov-report=term-missing
```

`--cov-branch` is an explicit diagnostic overlay; it changes no Gate policy.
The `-k wizard` selection is identical at both revisions, with exactly eight
additional passing cases after #2372. This is **measured wizard-selected
coverage**, not a full-suite Gate measurement or a crosswalk projection.
Environment: CPython 3.14.10; tool versions, exact executed commands, full output
paths, selection and settings are in
[coverage-comparison.json](./recheck/coverage-comparison.json).

| Wizard metric | Parent `10ac6839` | Merge `e8967edb` | Reduction |
| --- | ---: | ---: | ---: |
| Statement denominator | 1050 | 1050 | 0 |
| Branch denominator | 284 | 284 | 0 |
| Missing statements | 338 | 284 | **54** |
| Missing branches | 125 | 106 | **19** |
| Missing combined units | 463 | 390 | **73** |
| Passing selected cases | 108 | 116 | +8 |

Both runs exited 0; both deselected 1612 cases. The measured 73-unit reduction
exceeds the roughly 60-unit target **in this selection**. Inspect the unchanged
target-file entries, executed/missing lines and branches in
[coverage-base.json](./recheck/coverage-base.json) and
[coverage-merged.json](./recheck/coverage-merged.json), and complete output in
[base-wizard-coverage.txt](./recheck/base-wizard-coverage.txt) and
[merged-wizard-coverage.txt](./recheck/merged-wizard-coverage.txt).

Full-suite reconstruction remains **UNKNOWN**: identical broad runs with
`-m 'not slow and not live_llm' --cov --cov-config=pyproject.toml --cov-branch
-n 2 --dist=loadgroup` reported failures and stopped progressing near completion.
They were interrupted (exit 130); no terminal JSON was recovered. Commands and
limitations are in the comparison record; partial progress is retained in
[base-coverage.txt](./recheck/base-coverage.txt) and
[merged-coverage.txt](./recheck/merged-coverage.txt). Their failure causes were
not diagnosed. These runs used the current interpreter/dependencies, rather
than a recovered historical CI environment. No production change, exclusion,
test skip or dependency change was made to force them green.

Coverage boundaries: Gate measures product **lines**, not the dispatch's claimed
blended total; the supplemental branch overlay has denominator 1334 for this
target. Neither repository-wide 90% acceptance nor a historical full-suite
after value is inferred from the focused comparison. The original alternative
acceptance path would require a concrete explanation of why each remaining
target could not meaningfully be tested; no such impossibility was established
for the remaining wizard branches, so that alternative is **not claimed**.
The older 321-unit cumulative reduction and 54-line projection remain separate.

### Configured-suite attribution: paired CI (2026-11-10)

Stop rule recorded before changes: two paired-CI attempts, 45 minutes. Run
[42874673712](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/42874673712)
failed: the scalar `$TEST_COMMAND` split `-m 'not live_llm'` into `'not` and
`live_llm'`, project dependencies were never installed, and pytest collected
0 items (exit 5). Attempt 1,
[42887268107](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/42887268107),
installed the 254 historical packages but its environment check compared the
installer bootstrap tools `pip` and `uv`. Attempt 2,
[42887512783](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/42887512783)
at `135c022a`, completed successfully:
[parent job](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/42887512783/job/130079659001),
[merge job](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/42887512783/job/130079659222).

Each job verified its checkout SHA and `pyproject.toml` hash
`b442a73d…`, then installed dependencies through the historical mechanism
(`uv pip install --system -r requirements.lock -e .[app,dev]` plus the
checkout's `autofix-versions.env` test pins), forced to the 253-package set
resolved by historical parent job
[117210972329](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/38971176069/job/117210972329)
([resolved-overrides.txt](./recheck/paired-ci-input/resolved-overrides.txt)).
Both actual environments match that set exactly (CPython 3.12.15, uv 0.13.1,
pytest 9.1.1, pytest-cov 7.1.0, coverage 7.16.2, xdist 3.8.0, 4 workers).
Disclosed differences from history: the historical merge job resolved
`app-baseline-kit` 0.2.0 from Workflows `adb78bb`, the replay uses `2f9eb2d`
for both sides; the runner image is ubuntu26 20261101.186.1. Pytest ran
bounded (`timeout 25m`; job limit 45 minutes) with the command passed as a Bash
array:

```sh
pytest tests/ -m 'not live_llm' --junitxml=pytest-junit.xml --cov \
  --cov-config=pyproject.toml --cov-branch --cov-report=json:coverage.json \
  --cov-report=term-missing -n auto --dist=loadgroup
```

| Wizard target | Parent `10ac6839` | Merge `e8967edb` | Reduction |
| --- | ---: | ---: | ---: |
| Suite result | 1716 passed, 2 skipped | 1724 passed, 2 skipped | +8 |
| Statement / branch denominators | 1050 / 284 | 1050 / 284 | 0 |
| Missing statements | 338 | 284 | **54** |
| Missing branches | 125 | 106 | **19** |
| Missing combined units | 463 | 390 | **73** |

Suite counts and missing statements equal the historical branch=false CI logs,
which cross-checks the replay. Raw statement and branch identities are
identical in every measured file; repository totals fall from 3741 to 3668
combined units, so the whole reduction is in the target. The **isolated
combined reduction in the historical configured suite is 73 units**, above the
roughly 60-unit target. Receipts (raw coverage JSON, JUnit, terminal output,
commands, checkout and config hashes, actual environments, exit codes) are in
[recheck/paired-ci/](./recheck/paired-ci/) and bound in
[coverage-comparison.json](./recheck/coverage-comparison.json). The verifier now
rejects a mismatched checkout, config hash, split marker, missing `--cov-branch`,
differing command or environment, non-zero exit, failing or empty JUnit, foreign
or mixed run/job URLs, coverage outside the receipt, and any differing raw
file boundary. The focused 73-unit wizard result above remains supplemental.

### Ranking: permitted UNKNOWN disclosure

[ranking-search.json](./recheck/ranking-search.json) records searched issue and PR
records, linked Gate artifact listings and local evidence. Both inspected Gate
listings expose only a LangSmith artifact, with no dispatch ranking receipt.
The issue's 1/1/338 claim has no matched dispatch receipt or incident mapping in
these sources. Original provenance is **unrecoverable/UNKNOWN** in this bounded
search. The older #2336 ranking archive is not a substitute. This satisfies the
requested disclosure alternative and does **not** claim original selection
compliance.

### Runnable acceptance result

[contract.json](./recheck/contract.json) is the machine-readable counterpart to
the contract recorded above. Run the verifier from the repository root:

```sh
python scripts/verify_issue_2360_evidence.py
```

It requires inspectable receipts and checks revision binding, executed pytest,
all eight named red/green proofs and restoration hashes, comparable measured
coverage, explicit ranking disclosure, and the dated finite contract. Missing
or invalid mandatory proof produces `FAIL` and exit 1. The ranking disclosure
criterion permits UNKNOWN provenance; that permission never applies to CI or
individual mutation proofs. `OVERALL PASS` means these evidence requirements
pass, with the selection boundaries and UNKNOWN findings stated above; it does
not assert independent ranking compliance or reconstructed full-suite coverage.

Actual verification output: [verification-output.txt](./recheck/verification-output.txt),
exit **0**, all six evidence checks PASS. Additional validation is recorded in
[validation.json](./recheck/validation.json): 34 targeted tests pass, verifier
statement coverage is 97%, Ruff passes, and Black's complete repository check
leaves 535 files unchanged. Default Black multiprocessing cannot create its
local forkserver socket in this sandbox; the passing check used one worker and
a temporary asyncio wake-up timer, with the same formatting rules and CLI
arguments. No formatter configuration or dependency was changed.

Delivery limitation: staging the verified source and receipts with `git add`
exited 128 because `.git/index.lock` cannot be created on the checkout's
read-only Git metadata. This run cannot advance the working checkout's branch.
A source commit and importable patch can be prepared in a disposable clone
under writable `/tmp`; that does not change the assessed historical revisions.

## Configured-suite attribution continuation (2026-11-10)

Outcome 1 before changes: configured-suite attribution UNKNOWN. The existing
73-unit wizard-selected result is supplemental and cannot satisfy #2360's Gate
configured-suite requirement. Earlier OVERALL PASS statements above describe
historical verifier behavior and are superseded for this requirement.

Acceptance command: `/opt/anaconda3/bin/python3 scripts/verify_issue_2360_evidence.py`.
The outcome is requirement-by-requirement PASS/FAIL/UNKNOWN, with OVERALL FAIL
(exit 1) while configured-suite attribution is unavailable. Focused regression:
`/opt/anaconda3/bin/python3 -m pytest -q -o addopts= tests/test_issue_2360_evidence.py`.
No local virtual environment is selected or present in this checkout; use the
owner-selected fallback interpreter. Check date: 2026-11-10.

Stop rule recorded before changes: two retrieval attempts per historical receipt,
20 minutes for historical recovery, 45 minutes for this continuation excluding
the required post-push observation floor. No local full-suite execution: the
owner explicitly reserves that to CI. Recover historical Gate artifacts/logs at
the adjacent parent and original merge, inspecting isolated checkouts' settings.
If terminal comparable raw coverage is absent, stop recovery, keep UNKNOWN and
OVERALL FAIL, and deliver the verifier correction without asserting attribution.
Blocking/drainable: missing configured-suite receipts block acceptance; focused
regression, honest evidence recording and PR delivery remain drainable.
No production changes, dependency changes or repeated mutation/ranking work.

### Historical recovery result

Isolated detached checkouts were created at `/tmp/issue2360-configured-parent`
and `/tmp/issue2360-configured-merge` from a shared disposable clone. All three
clone/worktree commands exited 0. Both exact heads and identical pyproject hashes
are recorded in `recheck/coverage-comparison.json`. Only the original eight-case
test file differs; no later tests were copied. Historical pytest defaults select
`tests/`, excluding `live_llm` only; **slow tests are included**. Both CI jobs used
blank extra markers/arguments and the same coverage sources and configuration.

Recovered main CI run `38971176069`, job `117210972329` at the adjacent parent,
and run `38977412115`, job `117230894905` at the original merge. The run/job API
receipts and timestamped command/environment/terminal excerpts are stored as
`recheck/configured-{parent,merge}-*`. `gh api .../actions/jobs/<job>/logs` exited
0 for each. Both pytest jobs succeeded: **1716 passed, 2 skipped** before and
**1724 passed, 2 skipped** after. The merge run failed its separate format job;
it is not claimed as overall successful CI. No Gate workflow run is indexed at
either exact main SHA (both workflow-specific queries returned zero runs).
These configured CI records are not relabeled as PR Gate records.

The logged executed command is:

```sh
pytest --junitxml=pytest-junit.xml --cov --cov-config=pyproject.toml \
  --cov-report=xml:coverage.xml --cov-report=term-missing \
  --cov-report=json:coverage.json -n auto --dist=loadgroup
```

Environment: Linux, CPython 3.12.15, pytest 9.1.1, pytest-cov 7.1.0,
xdist 3.8.0, two workers. Both records print 1050 wizard statements, with
**338 missing before and 284 after: 54 fewer uncovered statements**.
Historical `branch=false` is explicit in both isolated pyprojects. No combined
statement-plus-branch delta can be calculated. In particular, the focused
19-branch improvement cannot be added to this configured-suite line result.

Run artifact inventories contain only the LangSmith artifact. Logs name original
coverage artifacts `12077805365` and `12080251116`, but direct metadata queries
for both exited 1 with HTTP 404. Raw executed/missing line and branch identities
are unavailable, recorded as null. Literal compressed missing-line ranges from
the logs are preserved; ranges must not be expanded into invented executable
line identities. The recovered metadata and excerpts identify the exact source
URLs/commands, exits and environment. No local full-suite run was attempted.

The original alternative is **not substantiated**: the sole wizard target was
meaningfully testable, as the preserved eight mutation proofs demonstrate.
Unavailable coverage receipts do not establish an untestable target. Mandatory
configured-suite combined attribution remains **UNKNOWN**, and the verifier
must print **OVERALL FAIL**. The unmodified focused receipt is nested under
`supplemental_focused`, preserving its commands, settings, exits and raw coverage
file paths and its valid 73-unit result. Existing CI, mutation, ranking and raw
coverage files are unchanged.

### Repair validation

Before the fix, the focused-receipt regression printed `PASS: coverage
attribution: isolated wizard selection: 73 fewer statement-plus-branch units;
not Gate` and `OVERALL PASS`, incorrectly returning 0. The focused run reported
2 failed, 59 passed: that new regression plus the updated default-UNKNOWN
expectation. The repair now rejects focused selection even if its scope label
is changed. Synthetic configured-selection metadata is confined to temporary
test fixtures so existing raw-unit corruption/shortfall regressions still test
their intended checks; it is never published as measured evidence.

Final focused validation (owner-selected `/opt/anaconda3/bin/python3`, CPython
3.12.2, pytest 9.2.0): `-m pytest -q -o addopts=
tests/test_issue_2360_evidence.py` exited 0: **62 passed, 2 warnings in 3.08s**.
Warnings concern installed numexpr/bottleneck versions; pytest also prints an
asyncio fixture-scope deprecation. No dependency pin was changed.
`scripts/verify_issue_2360_evidence.py` exited **1**, printing five PASS results,
**UNKNOWN: coverage attribution**, and **OVERALL FAIL**. Actual stdout replaces
`recheck/verification-output.txt`. `git diff --check` passed. Outcome 1 remains
UNKNOWN; configured-suite combined attribution is left undone. Historical
retrieval stopped after the completed queries and both coverage-artifact 404s;
no further local execution or invented alternative is warranted.

## Paired-CI execution contract (2026-11-10)

Finite paired-CI execution contract for issue #2360 attribution at historical
revisions `10ac6839518576d4dc5199a24392eadfc388c713` (squash parent) and
`e8967edb2aa1bdbc1a68ea58ed241cf205915d03` (PR #2372 merge).

Workflow: `.github/workflows/issue-2360-paired-ci-attribution.yml`

Contract terms:
- **Revisions**: Two exact commits checked out separately in isolated environments
- **SHA verification**: Both `git rev-parse HEAD` and pyproject.toml SHA-256 hash
  `b442a73dc1a9c83e895e6613a07f56d2604799da127ce14edee20adaf55d06d4` verified
- **Dependencies**: Identical recorded dependencies installed via `pip install pytest pytest-cov pytest-xdist`
- **Selection**: `tests/ -m 'not live_llm'` (includes slow tests, excludes live_llm only)
- **Command**: `pytest tests/ -m 'not live_llm' --cov --cov-config=pyproject.toml --cov-branch --cov-report=json:coverage.json --cov-report=term-missing -n auto --dist=loadgroup`
- **Environment**: Ubuntu Linux, Python 3.12.15, pytest 9.1.1, pytest-cov 7.1.0, xdist 3.8.0, 2 workers
- **Output**: Raw coverage JSON, terminal output, and metadata for both revisions
- **Target**: About 60 fewer uncovered statement-plus-branch units measured
- **Bounded execution**: Workflow triggered on branch push or manual dispatch only
- **Artifact retention**: 90 days

Expected outcomes:
- Parent (10ac6839): ~1716 passed, 2 skipped, coverage JSON with branch data
- Merge (e8967edb): ~1724 passed, 2 skipped, coverage JSON with branch data
- Combined delta: Measured against 60-unit target using matching raw statement and branch boundaries

Receipt storage: Terminal job receipts, executed commands, environments and raw
coverage JSON stored under `docs/evidence/issue-2360-closer-repair/recheck/` as
`coverage-{parent,merge}-branch.json` and related metadata files.

Stop rule: Two retrieval attempts per historical receipt; 45 minutes total for this round.
No local full-suite execution permitted per owner constraint.

Acceptance: Workflow execution establishes comparable statement-plus-branch boundaries.
Verifier updates to accept branch coverage attribution when raw coverage JSON files
and workflow receipts are present and validated.

Current status: Workflow file committed. Branch coverage measurement pending CI execution.
Historical statement-only attribution (54 fewer uncovered statements) remains documented
separately and does not satisfy the configured-suite branch requirement.

### Paired-CI outcome (2026-11-10)

The contract above was executed with two corrections: the dependency install
uses the historical uv mechanism and the recorded 253-package resolved set (not
the bare `pip install pytest pytest-cov pytest-xdist` above, which installed no
project dependencies), the runner used 4 xdist workers, and receipts are stored under
`recheck/paired-ci/<label>/` rather than the file names listed above. Run 42887512783
succeeded on both sides; see "Configured-suite attribution: paired CI
(2026-11-10)". `python scripts/verify_issue_2360_evidence.py` now exits 0 with
six PASS results, `PASS: coverage attribution: configured suite: 73 fewer
statement-plus-branch units`, and `OVERALL PASS`. The earlier UNKNOWN/OVERALL
FAIL statements are retained as the dated record of the state before this run.

### Configured-suite receipt verification follow-up (2026-11-10)

Before source/test changes, inspected `69ad80d8` and the historical parent/merge
diff: only the original eight-case test file changes. The retained paired run
already supplies the missing measurement; baseline verification prints six PASS
results and OVERALL PASS, and all 76 evidence tests pass. Live connector reads
of #2360, #2372, #2467 and run 42887512783's jobs agree with retained identities
and successful checkout/execution steps. `gh pr view 2372` exited 1 because
shell network access is unavailable.

This follow-up verifies that the actual recorded command cannot narrow the
configured selection on both sides, and that each job is bound to the retained
successful paired run and its exact historical checkout. Run
`python scripts/verify_issue_2360_evidence.py`; expect six PASS results and
OVERALL PASS for unchanged receipts, FAIL and exit 1 for narrowed selection or
contradictory run/job evidence. Stop after two attempts per unavailable remote
receipt or 45 minutes total. Reuse the completed paired execution rather than
repeat historical suites; preserve focused, historical and ranking disclosures.

Verified result: all 19 new regression cases failed before the fix; afterward,
`python -m pytest tests/test_issue_2360_evidence.py tests/test_wizard_render_coverage.py
-q -m 'not slow' --cov=scripts.verify_issue_2360_evidence --cov-report=term-missing`
exited 0 with **112 passed** and **98%** verifier coverage. The acceptance command
prints six PASS results and OVERALL PASS (exit 0); Ruff and `git diff --check`
pass. Black formatted the changed Python files individually; the required full
repository check passed with **535 files unchanged**. As in previous rounds,
the sandbox rejects Black's forkserver socket and the plain single-worker run
stalled; a temporary standard asyncio loop with a 50 ms wake-up timer and one
worker completed the same Black command without changing formatting settings.
See `configured_suite_binding_followup` in [validation.json](./recheck/validation.json).

- [x] Inspect configured pytest/coverage commands and verify comparable historical
  coverage at adjacent parent `10ac6839` and original merge `e8967edb`, excluding
  later tests. Both use 1050 statements and 284 branches in the target; missing
  combined units fall from 463 to 390, a **73-unit** measured reduction.
- [x] Configured-suite statement-plus-branch attribution measured at adjacent
  parent/original merge. Retained paired run 42887512783's successful jobs and
  raw coverage establish attribution independently of the supplemental slice.
- [x] The recorded configured command is runnable on both assessed revisions;
  isolated measured reduction exceeds the roughly 60-unit target. Original
  mutation proofs, historical cumulative gain and ranking UNKNOWN stay separate.

The checkout's `.git` directory is read-only. Deliver this follow-up as a source
commit in `/tmp/issue2360-configured-suite-delivery` and an importable patch at
`/tmp/issue2360-configured-suite.patch`; this does not advance the checkout's
branch. PR #2485 was inspected open with `draft=false` at `69ad80d8`.

The verified PR-body reconciliation (three current unchecked tasks, preserving
the nested historical issue) was attempted after validation and rejected:
`MCP tool call requires approval, but approval policy is never`. Remote task
checkboxes and the obsolete PR-body UNKNOWN outcome remain unchanged. This is
a publication blocker, separate from the verified local acceptance result.

## Executed-case attribution contract (2026-11-10 UTC)

Before source/test edits, reviewed commits `bd531f3d`, `efc96609`, and
`69ad80d8`, both historical pyprojects via `git show`, the adjacent revision
diff (only the original eight-case file), and both retained JUnit reports.
Live connector run 42887512783 confirms both exact historical checkout and
execution jobs completed successfully. Baseline: 103 evidence tests pass; the
acceptance command prints six PASS results and OVERALL PASS. The configured
receipts already satisfy both remaining coverage checkboxes: 1050 statements
plus 284 branches, 463 to 390 uncovered units, measured reduction 73.

This follow-up binds that attribution to executed case identities: the merged
JUnit report must contain precisely the parent cases plus the eight original
successful cases, excluding later tests. Duplicate, skipped, failed, missing,
or unrelated replacement cases must prevent coverage PASS. Run
`python scripts/verify_issue_2360_evidence.py`; expect six PASS results and
OVERALL PASS for retained receipts, coverage FAIL and exit 1 for contradictory
case receipts. Stop after two retrieval attempts per unavailable receipt or
45 minutes total. Reuse the completed paired CI; no local historical full-suite
replay or new measurement is required.

Before continuing, reconciled the two stale current PR #2485 checkboxes and
obsolete top-level UNKNOWN result in a proposed body, preserving historical
nested issue text. Publication was rejected: `MCP tool call requires approval,
but approval policy is never`. Remote checkboxes remain unchanged. The PR was
read open with `draft=false`. This is a publication limitation, separate from
the retained measurements.

Verified outcome: all ten new regression cases failed before implementation
(nine contradictory receipts incorrectly passed; malformed XML escaped the
verifier). After implementation, **130 targeted tests pass**, and the verifier
module reports **98%** statement coverage (175 statements, 4 missing). The
retained JUnit reports contain **1718 parent / 1726 merged cases**, with two
skips each; all eight added original cases pass and all shared case identities
and skipped statuses match. Acceptance prints six PASS results and OVERALL PASS.
Ruff and `git diff --check` pass. Black formatted the two changed Python files
individually; its full repository check passed with **535 files unchanged**.
As in prior rounds, that check required one worker and a temporary standard
asyncio loop with a 50 ms wake-up timer after default multiprocessing was
restricted and the plain one-worker run timed out. See
`executed_case_attribution_followup` in [validation.json](./recheck/validation.json).

- [x] Verify configured-suite attribution using both historical raw coverage
  records and exactly the original eight executed added cases.
- [x] Verify the remaining coverage acceptance criterion: identical target
  boundaries, uncovered counts 463 to 390, isolated measured reduction 73.

The main checkout cannot stage changes: `git add` exited 128 because its
`.git/index.lock` is read-only. The source commit is prepared in
`/tmp/issue2360-executed-cases-delivery`, with an importable patch at
`/tmp/issue2360-executed-cases.patch`. The reconciled PR body is saved at
`/tmp/issue2360-executed-cases-pr-body.md`; the rejected remote update is not
claimed as published. No workflow or production file changed.
