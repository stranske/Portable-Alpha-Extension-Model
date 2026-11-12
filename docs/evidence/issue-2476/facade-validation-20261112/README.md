# Mixed-row facade validation, 2026-11-12 UTC

Baseline: `6e6fe89111f16771a1152b5c8f719be861550c88`.
Measured source commit: `28a3d750bd157cfdff216ea9dcc3c67796daf87f`.
Available main: `5be77e15c925f5354ab67e2a1749fe6187d2038c`.
Python: `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python`, version 3.14.10.
The evidence follow-up changes no Python files.

## Reconciliation before edits

Read the recent commits, issue-format and target-environment guides, the
candidate review of the complete #2473 diff, the hosted main/merge receipts,
and the latest helper verification. The pre-edit candidate run passed **91
tests in 6.41s**. The last source commit's four helper mutations and its
coverage timeout were already preserved in `verification-20261112/`.

Attempted to update PR #2498 before continuing. Automatic approval review
rejected the body update: `MCP tool call requires approval, but approval
policy is never`. No remote checkbox was changed. The proposed verified
reconciliation is in [reconciliation.md](reconciliation.md). The PR was open
with `draft=false`, remote head matching this baseline.

## First incomplete task: bounded baseline attempt

Repeated the requested `rg -n 'coverage|pytest|lint-format|Gate' .github
pyproject.toml` inspection, excluding pre-existing untracked node_modules on
subsequent reads. Consumer `ci.yml` delegates to Workflows. The current
reusable workflow is blob `f2a1abdccc0db5218206917b9e8b45d11b636717`, the
same blob recorded by the previous continuation. `pyproject.toml` measures
product directories with `branch=false`; the separate focused diagnostic
adds `--cov-branch`.

[contract.json](contract.json) records the pre-edit baseline, all seven
ranked paths, configuration hash, desired behavior, commands and stop rule.
Allow one full-suite attempt per revision/configuration, capped at 180s;
focused acceptance and each mutation are capped at 120s. Do not infer a
coverage delta from a timeout. Stop a failing step after three attempts.

The local baseline command was:

```sh
timeout -k 5s 180s python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml:/tmp/issue-2476-current/baseline.xml --cov-report=term-missing --cov-report=json:/tmp/issue-2476-current/baseline.json -n auto --dist=loadgroup -m 'not slow and not live_llm' -vv
```

It exited **124**, without a coverage report. The [complete log](baseline-gate.txt.gz)
identifies **2,058 completed cases**; only
`tests/test_cli.py::test_main_with_png` started without a completion line.
No failed case was reported before timeout. The renderer root cause is not
established. A head full-suite run was not attempted after the incomplete
baseline. This command retains the coverage flags and scheduler but omits
the current hosted workflow's generated `ci_test_growth` observer and JUnit
output; it is not an exact reproduction of current hosted Gate execution.

The first incomplete coverage task remains open. Proceeded to the next
task's behavioral assertions using the existing hosted saturation evidence.

## Ranked review and measured contribution

The prior [hosted main/merge receipts](../continuation-20261111/hosted-main/seven-targets.json)
show target 1 saturated, target 2 saturated except the unreachable argparse
exit, and target 3 saturated except two unreachable export statements/arcs.
Targets 4–6 are saturated there. All seven production files still match the
hosted candidate `4b1f5e4c` exactly. Target 3's actual mixed-row production
fix remains worth checking against both container types and row orders.
No further target was skipped on a new saturation claim.

Ran identical six-module focused coverage scopes before and after the test
change. [Baseline](focused-baseline.json) passed **153 cases in 8.37s**
(10.591s wall); [head](focused-head.json) passed **156 cases in 8.12s**
(10.323s wall). Exact commands, revisions, input hashes and coverage metadata
are in the extracts. The full reports are the adjacent `.coverage.json.gz`
files; their uncompressed hashes are recorded in the extracts.

| Rank / target | Statements | Missing before → after | Branches | Missing before → after |
| --- | ---: | ---: | ---: | ---: |
| 1. `pa_core/llm/result_explain.py` | 227 | 0 → 0 | 102 | 0 → 0 |
| 2. `pa_core/pa.py` | 222 | 19 → 19 | 68 | 10 → 10 |
| 3. `pa_core/facade.py` | 337 | 15 → 15 | 124 | 14 → 14 |
| 4. `pa_core/units.py` | 81 | 3 → 3 | 30 | 3 → 3 |
| 5. `pa_core/sim/internal_pa_financing.py` | 35 | 0 → 0 | 18 | 0 → 0 |
| 6. `pa_core/manifest.py` | 61 | 2 → 2 | 4 | 2 → 2 |
| 7. `dashboard/pages/4_Results.py` | 210 | 0 → 0 | 70 | 1 → 1 |

The focused scope does not establish saturation for targets 2–4 or 6.
Results' historical 136 missing units are not its current count: this scope
has zero missing statements and the invariant exception arc 178→180.
[deltas.json](deltas.json) confirms **zero new coverage units**. The change
strengthens rejection and input-preservation assertions.

Round attribution remains unchanged: historical #2384 removed 58 statements
and 14 branches, intervening work removed 131/79, and the hosted main/merge
PR step removed 18/26. The PR's measured contribution is 44 units; the
historical round's 326 units exceed ~180 only under that historical policy.
The matched-policy round baseline remains unmeasured after the earlier
dependency-installation conflict. The reason #2384 selected target 7 first
remains unknown. No round-completion claim is made here.

## Actual production mutations

`test_serialize_agent_semantics_mixed_series_rejects_duplicate_labels` now
covers Series-first and dict-first inputs in lists and tuples. Rejection
must retain the original container, rows, Series labels/values/name and
mapping contents. Four actual production mutations each failed all four
named cases and passed all four after restoration:

| Mutation | Exact production diff | Named failures | Restored pass |
| --- | --- | --- | --- |
| Remove duplicate-label guard | [diff](accept-duplicates.diff) | [red](accept-duplicates-red.txt) | [green](accept-duplicates-green.txt) |
| Replace rejected input | [diff](replace-rejected-input.diff) | [red](replace-rejected-input-red.txt) | [green](replace-rejected-input-green.txt) |
| Alter caller's Series | [diff](alter-series-before-rejection.diff) | [red](alter-series-before-rejection-red.txt) | [green](alter-series-before-rejection-green.txt) |
| Clear caller's mapping | [diff](alter-mapping-before-rejection.diff) | [red](alter-mapping-before-rejection-red.txt) | [green](alter-mapping-before-rejection-green.txt) |

[mutations.json](mutations.json) retains runnable commands, named failures,
timings and hashes. Every red exit is 1, every restored exit is 0, and
before/restored production hashes match. Restoration occurred in `finally`
before every green run. `git diff --exit-code -- pa_core/facade.py` passed.
There is no production change.

## Lint, acceptance and delivery limits

The original lint-format failure is preserved in
[the archived job output](../measurement-20261111/original-lint-format.txt):
run 39044091242, job 117429268297 requested Black formatting of
`tests/test_wizard_render_coverage.py`. Main fixed it in `d82092b7` (#2397).
Its bytes still match available main. No repair to that file is attributable
to this continuation. The reusable lint-format job runs Black; PR Gate
sets `format_check: false`, so its skipped formatting job is not a pass.

- [Requested Results acceptance](focused-results.txt): **60 passed in 2.86s**, **4.81s wall** using `timeout -k 5s 120s /usr/bin/time -p python -m pytest tests/test_coverage_round_2381.py -m 'not slow and not live_llm' -q`.
- [Combined acceptance](focused-acceptance.txt): **94 passed in 3.18s**, running the Results, facade, evidence and mutation-transcript modules with `-m 'not slow and not live_llm'`.
- [Ruff](ruff.txt): `ruff check .` passed.
- Black formatted the modified file at line length 100. The exact required repository command passed with **538 files unchanged**: `BLACK_NUM_WORKERS=1 PYTHONPATH=/tmp/issue-2476-black-runtime timeout -k 5s 45s black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .`. The [runtime shim](black-runtime.py.txt) is the previously recorded sandbox workaround, installed as `sitecustomize.py` only in that temporary directory. It selects fork and polls event-loop completions; Black's discovery, formatting and safety checks remain active. The [plain command's socket permission failure](black-plain-failure.txt) and [passing check](black-check.txt) are preserved separately.
- Required hosted CI and comparable complete baseline/head Gate reports remain outstanding. No passing result is claimed for the delivered source/evidence commits.

The normal `git add` failed because `.git/index.lock` is on a read-only
filesystem. Source commit `28a3d750` exists in
`/tmp/issue-2476-current/delivery.git`; an evidence-only follow-up and a
bundle preserve delivery for a writable checkout. The checkout's HEAD
remains `6e6fe891`; its working-tree edits are available to automation.
No push or workflow edit was performed.
