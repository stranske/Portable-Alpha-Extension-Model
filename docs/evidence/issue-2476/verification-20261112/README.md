# Results helper verification, 2026-11-12 UTC

Baseline: `6c919c92c99049c6672256a7e1aa7ee3fb1189b1`.
Measured source commit: `bda7804e12971463fe8d28a2186759ee3f8e6411`.
Available main: `5be77e15c925f5354ab67e2a1749fe6187d2038c`.
Python: `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python`, version 3.14.10.
The evidence follow-up changes no Python files.

## First unchecked task and stop rule

Before changing tests, read the issue-format and target-environment guides,
the existing candidate review, the measurement contracts, and the current
Results tests and helper implementations. Ran the requested
`rg -n 'coverage|pytest|lint-format|Gate' .github pyproject.toml` search;
subsequent searches excluded the pre-existing untracked node_modules.
[contract.json](contract.json) records the exact baseline, seven target paths
in dispatch order, runnable coverage command, configuration hash and stop rule.

One full-suite attempt per revision/configuration is allowed, capped at 180
seconds; focused acceptance is capped at 120 seconds. Stop a failing mutation
step after three attempts. Never infer a coverage delta from a timeout.

The pre-edit command retained the Gate coverage flags from the archived
discovery, changing only report destinations:

```sh
timeout -k 5s 180s python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml:/tmp/issue-2476-20261112/baseline.xml --cov-report=term-missing --cov-report=json:/tmp/issue-2476-20261112/baseline.json -n auto --dist=loadgroup -m 'not slow and not live_llm'
```

It exited **124** without a coverage report. The [complete progress log](baseline-gate.txt)
prints a failure marker at 38% and reaches 97% before timing out. The failed
and unfinished test identities and the timeout cause remain **unknown**.
No head full-suite run was attempted after that incomplete baseline.

The current upstream reusable workflow was also inspected through GitHub:
blob `f2a1abdccc0db5218206917b9e8b45d11b636717`.
[workflow-inspection.json](workflow-inspection.json) retains relevant lines.
It keeps the same coverage configuration/report flags and xdist scheduler,
but now adds JUnit output and a generated `ci_test_growth` observer loaded by
`pytest -p ci_test_growth`. That observer was absent from the local command
above. This is coverage-configuration evidence, not an exact reproduction of
the current hosted Gate. Neither a passing full-suite baseline/head comparison
nor completion of the first unchecked task is claimed.

## Ranked targets and comparable focused artifacts

The pre-edit diagnostic passed **152 cases in 8.23s**. The head passed
**153 cases in 8.02s**. Both run the same six test modules with the same product
coverage configuration and `--cov-branch`, under a 120-second timeout. The
extra case separates two exception handlers within an existing test.

[Baseline](focused-baseline.json) and [head](focused-head.json) retain exact
revisions, commands, input hashes, configuration hashes, coverage metadata,
totals and missing lines/arcs for every ranked target. Full coverage JSON is
preserved in the adjacent `.coverage.json.gz` files, with uncompressed SHA-256
values in the extracts. Their logs are `focused-baseline.txt` and
`focused-head.txt`. [deltas.json](deltas.json) records the comparison.

| Rank / target | Statements | Missing before → after | Branches | Missing before → after |
| --- | ---: | ---: | ---: | ---: |
| 1. `pa_core/llm/result_explain.py` | 227 | 0 → 0 | 102 | 0 → 0 |
| 2. `pa_core/pa.py` | 222 | 19 → 19 | 68 | 10 → 10 |
| 3. `pa_core/facade.py` | 337 | 15 → 15 | 124 | 14 → 14 |
| 4. `pa_core/units.py` | 81 | 3 → 3 | 30 | 3 → 3 |
| 5. `pa_core/sim/internal_pa_financing.py` | 35 | 0 → 0 | 18 | 0 → 0 |
| 6. `pa_core/manifest.py` | 61 | 2 → 2 | 4 | 2 → 2 |
| 7. `dashboard/pages/4_Results.py` | 210 | 0 → 0 | 70 | 1 → 1 |

This continuation removes **zero uncovered statements and zero branches**.
Stronger assertions do not earn a coverage reduction. The focused suite does
not establish saturation for targets 2–4 or 6. No new seven-target completion
is claimed.

The previous [hosted main/merge receipts](../continuation-20261111/hosted-main/seven-targets.json)
measured target 2 with no missing statements and one unreachable parser exit
arc, target 3 with two unreachable post-serialization statements/arcs, and
targets 1, 4, 5 and 6 with no gaps. All seven production files at this baseline
match candidate `4b1f5e4c` byte for byte (`git diff 4b1f5e4c HEAD --` the seven
paths was empty before the source commit). These prior receipts justify
reviewing target 7's helper assertions; they are not a fresh saturation claim
for this narrower test scope. Results' historical 136 uncovered units is not
its current focused count: this remeasurement has zero missing statements
and only arc 178→180. The unavailable-without-message fallthrough remains a
documented invariant exception.

The [historical round](../gate-coverage.json) accounts for 207 statements and
119 branches: #2384 contributed 58/14, intervening changes 131/79, and earlier
#2476 work 18/26. The previous current-policy main/merge receipts independently
confirm that PR step of **44 units**, not 326 units attributable to this PR.
The historical 326-unit round total exceeds the approximate 180-unit objective
only under its historical test policy. The matched-policy round baseline
remains unmeasured after the earlier dependency conflict; this continuation
does not resolve it. The reason #2384 originally selected target 7 first is
still unknown.

## Behavioral assertions and production mutations

`test_render_comparison_panel_shows_info_when_previous_run_unavailable` now
checks the exact missing-artifact message, returned availability/path, and
absence of renderer calls, errors and warnings. The explanation failure test
now runs separate `missing-llm` and `panel-error` cases and checks the exact
installation guidance. The former truthiness assertions accepted an empty
message because `[""]` is truthy.

Every strengthened case has an actual production mutation, a named failure,
and a passing rerun after restoration. [mutations.json](mutations.json)
records commands, timings, hashes and filenames. The diffs preserve exact
changed bytes with zero context; the complete outputs identify the failing
test nodes. Transcript lines have trailing whitespace removed for
`git diff --check`; original scratch outputs remain under `/tmp/issue-2476-20261112/`.

| Mutation | Exact production diff | Named failure | Restored pass |
| --- | --- | --- | --- |
| Empty comparison guidance | [diff](comparison-message.diff) | [red](comparison-message-red.txt) | [green](comparison-message-green.txt) |
| Render an unavailable comparison | [diff](comparison-unavailable-render.diff) | [red](comparison-unavailable-render-red.txt) | [green](comparison-unavailable-render-green.txt) |
| Empty missing-LLM guidance | [diff](explain-missing-message.diff) | [red](explain-missing-message-red.txt) | [green](explain-missing-message-green.txt) |
| Empty generic-error guidance | [diff](explain-error-message.diff) | [red](explain-error-message-red.txt) | [green](explain-error-message-green.txt) |

All red exits were 1; all green exits were 0. Production bytes were restored
in `finally` before each green run; before/restored hashes match in every
record. `git diff --exit-code -- dashboard/pages/4_Results.py` passed afterward.
There is no production change or claimed production bug fix.

## Validation and delivery limits

- [x] Requested focused acceptance: `python -m pytest tests/test_coverage_round_2381.py -m 'not slow and not live_llm' -q` passed **60 cases in 2.68s**, **4.696s wall time**. [Machine receipt](focused-results.json), [output](focused-results.txt).
- [x] Combined acceptance/evidence suite: **91 passed in 2.81s**. [Output](focused-acceptance.txt). The four modules are the Results round, facade, evidence, and mutation-transcript modules.
- [x] All four actual production mutations failed their named tests, followed by passing restored reruns.
- [x] Ruff repository check passed; [receipt](ruff.txt).
- [x] Black formatted the changed test at line length 100. The repository check passed with **538 files unchanged**, using the [existing sandbox launcher](black-sandbox-launch.py.txt): `BLACK_NUM_WORKERS=1 timeout -k 5s 45s python /tmp/issue-2476-20261112/black-sandbox-launch.py --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .`. [Output](black-check.txt). The plain required Black command failed when forkserver tried to bind a Unix socket; the one-worker command timed out. Both failures are preserved. The launcher only polls event-loop completions; it retains Black's discovery, formatter, safety checks and CLI flags.
- [ ] Passing comparable full-suite Gate baseline/head artifacts and matched-policy historical round attribution remain outstanding.
- [ ] Required hosted CI on the delivered source/evidence head remains unverified.

The original lint-format failure was inspected in the
[archived job output](../measurement-20261111/original-lint-format.txt): run
39044091242, job 117429268297, Black requested formatting of
`tests/test_wizard_render_coverage.py`. Commit `d82092b7` (#2397) fixed that
file. Its current bytes match main; it was not changed here. Consumer `ci.yml`
delegates lint-format to Workflows, while PR Gate sets `format_check: false`.
A skipped Gate formatting job does not count as a Black pass.

The normal `git add` and `git commit` failed because `.git/index.lock` is on a
read-only filesystem. A writable bare Git directory under
`/tmp/issue-2476-20261112/delivery.git` holds the meaningful source commit and
the evidence follow-up. `issue-2476-verification.bundle` in that same scratch
directory preserves both commits for a writable checkout. Working-tree edits
remain available to the automation system; the checkout's HEAD was not moved.
No push or PR mutation was performed. PR #2498 was inspected through the
GitHub connector: open, draft=false, remote head `6c919c92`. No required CI
result is claimed for the new local commits.
