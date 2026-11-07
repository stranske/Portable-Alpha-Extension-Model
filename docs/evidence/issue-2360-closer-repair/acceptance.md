# Issue #2360 acceptance recheck

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

## Initial inspection

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
