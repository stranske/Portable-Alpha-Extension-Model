# Ranked result-explanation assertions, 2026-11-12 UTC

Baseline: `5e3add596fb618367aa7aa4457773e17384f4199`.
Measured source: `bffd8914a83026c7b8b89f07da08e669b1cd9618`.
Available local main: `9d083d69796126c41d3a05aa3499cb4fe8cc11eb`.
Interpreter: `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python` (3.14.10).
The evidence follow-up changes no Python files.

## Reconciliation before edits

Read the issue-format and target-environment guides, recent source and evidence
commits, the existing Results/facade tests, the complete five-file #2473 diff
(`git diff 7613ffd9^ 7613ffd9`), its prior review, and the hosted main/merge
receipts. The issue's description of #2473 as unlanded is stale: its merge is
`7613ffd97a450b15e550f499b7386ec4e67f2c51`. Retained the existing tests after
the four-module candidate suite passed **94 cases in 4.46s**.

Attempted to reconcile all duplicate task/acceptance checkboxes in PR #2498
before further edits. Automatic approval review rejected that update:
`MCP tool call requires approval, but approval policy is never`. The proposed
body is saved in `/tmp/issue-2476-statistics/proposed-pr-body.md`; it was not
published. PR metadata reported open and `draft=false` at the baseline head.

Verified statuses, applied consistently to duplicate tasks locally:

- [x] Existing #2473 review and retained candidate test execution.
- [ ] Complete current Gate discovery/reduction task: configuration, rank order,
  commands and stop rules are recorded, but complete current Gate comparison
  remains blocked by the prior full-suite timeouts.
- [x] Existing baseline Gate measurements and historical selection reconciliation.
- [ ] Complete ranked test/exception round: this continuation strengthens target
  1's assertions and establishes no additional coverage reduction.
- [x] Existing metric labels/values, delegated calls and helper assertions.
- [x] Production-mutation evidence, including this strengthened test's six
  failures and restored passing reruns.
- [ ] Complete delivered-head Gate artifacts and matched-policy round accounting.
- [ ] Required hosted pytest and lint-format on the delivered head.

The specific Results acceptance command passes below two minutes. Its local
acceptance checkbox can be checked. Existing behavioral/mutation and demonstrated
production-fix acceptance evidence is retained; the complete-artifact, round-goal
and delivered-head CI acceptance checkboxes remain open.

## First incomplete task and stop rule

Repeated `rg -n 'coverage|pytest|lint-format|Gate' .github pyproject.toml`, then
restricted reads to workflow/configuration files to avoid pre-existing untracked
node_modules. `ci.yml` delegates to Workflows; the reusable workflow blob is
`f2a1abdccc0db5218206917b9e8b45d11b636717`. Gate uses product-only sources and
`branch=false`. The discovered coverage invocation is recorded in
[contract.json](contract.json), with the pre-edit baseline, ranked paths,
configuration hash, desired assertions and stop rule.

The previous two unchanged-configuration local baseline attempts timed out at
180 seconds without a coverage report. The last verbose attempt identified
`tests/test_cli.py::test_main_with_png` as the only started, unfinished case.
No renderer root cause is established. Do not repeat that full-suite attempt
without a changed condition, or infer coverage from it. Focused runs and each
mutation are capped at 120 seconds, with at most three attempts per failing
step. Proceeded to the next task using measured saturation evidence.

## Ranked review and changed assertion

Before editing, the identical six-module focused scope passed **156 cases in
9.85s**. Target 1 already covered **227/227 statements and 102/102 branches**.
Hosted [main/merge receipts](../continuation-20261111/hosted-main/seven-targets.json)
record target 2 saturated except its unreachable argparse exit, target 3 except
two post-serialization export statements/arcs, and targets 4–6 saturated. All
seven production files still match that hosted candidate. This focused scope
alone does not establish saturation for targets 2–4 or 6.

`test_analysis_output_contains_required_sections_and_is_json_serializable` in
`tests/test_result_explain.py` formerly checked only that sections were nonempty.
It now passes five rows with an unsorted, nondefault index through the public
entrypoint and checks row/column metadata, numeric count/mean/sample standard
deviation/min/max, all three quantiles and their values, the exact final three
records in input order, JSON round-tripping, and an unchanged input dataframe.
The statistics, quantile and sample helpers execute without stubbing.

The head's identical scope passes **156 cases in 10.20s**. The coverage table's
`pa_core/llm/result_explain.py` row reports **100%**, including branches.
[measurements.json](measurements.json) records both exact source revisions,
commands, input hashes, coverage metadata, full-report hashes and per-target
deltas. Complete reports are [baseline](baseline.coverage.json.gz) and
[head](head.coverage.json.gz). Both use `pyproject.toml` plus `--cov-branch`;
these are focused diagnostics, not full-suite Gate results.

| Rank / target | Statements | Missing before → after | Branches | Missing before → after |
| --- | ---: | ---: | ---: | ---: |
| 1. `pa_core/llm/result_explain.py` | 227 | 0 → 0 | 102 | 0 → 0 |
| 2. `pa_core/pa.py` | 222 | 19 → 19 | 68 | 10 → 10 |
| 3. `pa_core/facade.py` | 337 | 15 → 15 | 124 | 14 → 14 |
| 4. `pa_core/units.py` | 81 | 3 → 3 | 30 | 3 → 3 |
| 5. `pa_core/sim/internal_pa_financing.py` | 35 | 0 → 0 | 18 | 0 → 0 |
| 6. `pa_core/manifest.py` | 61 | 2 → 2 | 4 | 2 → 2 |
| 7. `dashboard/pages/4_Results.py` | 210 | 0 → 0 | 70 | 1 → 1 |

Results' historical 136 missing units are not its current count. This scope
has zero missing statements and only the invariant unavailable-comparison arc
178→180. Current full-suite exceptions remain attributed to the earlier
hosted receipts, including facade statements 751/757 and arcs 750→751/756→757,
and pa's arc 349→exit. No new target was skipped on an unsupported saturation
claim. The reason #2384 originally selected target 7 first remains unknown.

This continuation contributes **zero new coverage units**. Prior attribution
remains 58 statements/14 branches from #2384, 131/79 from intervening work, and
18/26 from this PR's hosted main/merge comparison. The PR's contribution is
44 units. The historical 326-unit round exceeds ~180 under its historical
policy; a matched-policy round baseline is still unmeasured after the recorded
dependency conflict. This continuation does not claim round completion.

## Production mutation verification

[mutations.md](mutations.md) retains exact production diffs, runnable commands,
the named failing test and complete red/green outputs. [mutations.json](mutations.json)
records exit codes, timings and before/mutated/restored hashes. Every mutation
exits 1; every restored rerun exits 0:

1. Return the first three rows instead of the last three.
2. Reduce the sample to two rows.
3. Omit standard deviation from numeric statistics.
4. Substitute quartiles for the 5th/50th/95th percentiles.
5. Return zero quantile values at the correct percentile keys.
6. Modify the caller's dataframe after preparing the payload.

Production bytes were restored in `finally` before each green run. The final
`git diff --exit-code -- pa_core/llm/result_explain.py` passes. No production
change was needed.

## Checks and delivery limits

[checks.txt](checks.txt) contains both focused coverage runs, the requested
Results acceptance command, formatting and lint output:

- Results acceptance: **60 passed in 3.42s**, **6.20s wall**, using
  `timeout -k 5s 120s /usr/bin/time -p python -m pytest tests/test_coverage_round_2381.py -m 'not slow and not live_llm' -q`.
- `ruff check .`: passes.
- Black formatted the changed Python file at line length 100. The required
  `black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .`
  passes with **538 files unchanged** using `BLACK_NUM_WORKERS=1` and the
  previously documented temporary `sitecustomize.py` workaround. Its full
  source and the plain command's multiprocessing socket permission error
  are preserved in the checks. It selects fork and polls event-loop
  completions; formatter discovery, formatting and safety checks remain active.
- The original lint-format failure requested formatting of
  `tests/test_wizard_render_coverage.py` (run 39044091242, job 117429268297).
  Main fixed it in `d82092b7` (#2397). This continuation does not edit it.
  The archived original output remains in `../measurement-20261111/`.
- Baseline hosted Gate run 43136295487 reports success for Python 3.14,
  Ruff, mypy and gate-summary, but **lint-format is skipped**. This is a
  baseline result, not CI evidence for the new source revision.

Normal `git add` failed because `.git/index.lock` is on a read-only filesystem.
Source commit `bffd8914` was created in
`/tmp/issue-2476-statistics/delivery.git`, with this baseline as parent.
An evidence-only follow-up and `/tmp/issue-2476-statistics/delivery.bundle`
preserve the commits for a writable checkout. The original checkout's HEAD
is unchanged and its working-tree edits remain available to automation.
No push or workflow edit was performed. Required hosted CI and comparable
complete baseline/head Gate reports remain outstanding.
