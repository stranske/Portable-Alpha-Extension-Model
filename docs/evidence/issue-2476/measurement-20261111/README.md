# Gate measurement continuation, 2026-11-11 UTC

The first incomplete task remains incomplete: the full Gate-configured local
baseline timed out at 180 seconds without a coverage report. Its verbose log
identifies one started test without a completion line:
tests/test_cli.py::test_main_with_png. That test invokes the real PNG exporter.
The renderer's root cause is not established. No tests were skipped or disabled
to manufacture a successful Gate measurement.

The complete output is in [full-baseline.log.gz](full-baseline.log.gz).
[contract.json](contract.json) records the exact baseline, seven ranked targets,
commands, configuration, ranking source, stop rule and outcome before test edits.
The reusable Workflows source blob was read directly:
e0e17babfda63e37739db2e82314ff97c835b1ad.
The repository-selected interpreter is Python 3.14.10. Both consumer workflows
delegate to that reusable Python workflow. Its Gate coverage flags are:

    python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml:coverage.xml --cov-report=term-missing --cov-report=json:coverage.json -n auto --dist=loadgroup -m 'not slow and not live_llm'

The explicit marker preserves the live-LLM exclusion and this run's required
slow-test exclusion. Locally, XML/JSON output goes to /tmp/issue-2476-measurement/;
verbose output identifies unfinished tests. The unchanged configuration measures
pa_core, dashboard, and design-system, with branch=false. The focused comparison
below adds --cov-branch as a diagnostic.

Stop rule: one full-suite attempt per revision/configuration, with a 180-second
limit; focused acceptance must finish within 120 seconds. Stop after three
attempts of a failing step. The known full-suite timeout was not repeated on
the head. Fixing the unrelated real-renderer integration test remains follow-up.

## Reconciliation and ranked work

Reviewed 55183acc (salvaged facade repair and tests) and def0eec0 (candidate
review, sidebar assertions and their production mutations). The pre-edit
candidate command from review-20261111.md, with both marker exclusions, passed
84 tests in 3.23 seconds. The existing four completed tasks retain their
historical evidence attribution. No additional full coverage task is complete.

Before editing tests, the five-module focused suite in ranked-baseline.json
passed 134 tests in 9.89 seconds. Target 1 has 227/227 statements and 102/102
branches covered, justifying advancement to target 2.

Strengthened test_pa_registry_list_and_get to check zero and absent seeds,
every field and record in list output, the ID forwarded to registry lookup,
and the complete returned JSON record. A zero seed must remain zero.

| Actual production mutation | Named failure | Restored rerun |
| --- | --- | --- |
| Treat zero as an absent seed | [red](zero-seed-red.txt) | [green](zero-seed-green.txt) |
| Omit list record code version | [red](list-fields-red.txt) | [green](list-fields-green.txt) |
| Forward the wrong lookup ID | [red](get-id-red.txt) | [green](get-id-green.txt) |
| Return only the scenario ID | [red](get-fields-red.txt) | [green](get-fields-green.txt) |

Every red command exited 1 in
tests/test_coverage_round_2381.py::test_pa_registry_list_and_get; every restored
command exited 0. [mutations.json](mutations.json) records exact commands,
mutations, timings and hashes; adjacent zero-context diffs identify each site.
pa_core/pa.py was restored byte for byte in finally before each green rerun.
There is no production change in this continuation.

## Comparable focused measurements

Baseline: def0eec035bdccfdebce62c0116aa468ed6fdd2f.
Measured source commit: d39cfa1e205f29d2cb2a104eda45a00d14def011.
The source commit changes only the registry test. The evidence follow-up leaves
those Python bytes unchanged.

[Baseline](ranked-baseline.json) and [head](ranked-head.json) use identical
configuration and test scope. They retain revision IDs, commands,
configuration/test hashes, coverage metadata, statement/branch totals, and
missing lines and arcs for all seven targets. Full output is in
the corresponding .txt files. Head: 134 passed, 11.52 seconds wall time.

| Rank / target | Statements | Missing before → after | Branches | Missing before → after |
| --- | ---: | ---: | ---: | ---: |
| 1. pa_core/llm/result_explain.py | 227 | 0 → 0 | 102 | 0 → 0 |
| 2. pa_core/pa.py | 222 | 34 → 34 | 68 | 14 → 14 |
| 3. pa_core/facade.py | 333 | 15 → 15 | 120 | 14 → 14 |
| 4. pa_core/units.py | 81 | 3 → 3 | 30 | 3 → 3 |
| 5. pa_core/sim/internal_pa_financing.py | 35 | 0 → 0 | 18 | 0 → 0 |
| 6. pa_core/manifest.py | 61 | 2 → 2 | 4 | 2 → 2 |
| 7. dashboard/pages/4_Results.py | 210 | 17 → 17 | 70 | 7 → 7 |

These are focused measurements, not whole-suite gaps. This subset does not
establish saturation for pa.py or facade.py. Historical full-suite records
describe zero missing statements plus one unreachable parser branch for pa.py,
and two unreachable post-serialization statements/branches for facade.py;
those claims remain attributed to their historical revisions.
Target 7's historical 136 is not its current count. The previous review's
different scope included previous-run tests, yielding zero missing Results
statements; that scope must not be compared with this table as a delta.
Targets 3–7 were not changed or declared complete here.

This continuation reduces uncovered statements and branches by zero.
Historical records attribute 207 statement and 119 branch reductions to the
round: 58/14 to #2384, 131/79 to intervening work, and 18/26 to the earlier #2476
implementation. Their combined 326 units account for the historical approximate
180-unit objective, but cannot replace a fresh full-suite comparison.
The original reason #2384 selected target 7 before targets 1–6 remains unknown.

## Validation and delivery

- [x] Focused acceptance: python -m pytest tests/test_coverage_round_2381.py -m
  'not slow and not live_llm' -q passed 52 tests in 2.54 seconds, 4.48 seconds wall.
  [Receipt](focused-acceptance.json), [output](focused-acceptance.txt).
- [x] Four production mutations failed and passed after restoration.
- [x] Ruff passed: ruff check --output-format github --extend-exclude .workflows-lib .
- [x] Black formatted the changed file at line length 100; its repository check
  passed with 538 files unchanged and the required exclusion flags.
- [ ] Fresh whole-suite Gate baseline/head coverage and required CI on the
  delivered source commit remain unverified.

Ordinary Black failed because Python's forkserver cannot bind a Unix socket in
this sandbox. The fork launcher then hit its 45-second limit. The final launcher
adds an asyncio polling timer to process completion notifications; Black's
discovery, formatter, safety checks and flags are unchanged.
[Exact launcher](black-sandbox-launch.py.txt), [passing output](black-check.txt).
Its command was:

    BLACK_NUM_WORKERS=2 BLACK_CACHE_DIR=/tmp/issue-2476-measurement/black-cache timeout -k 5s 45s python /tmp/issue-2476-measurement/black-sandbox-launch.py --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .

The original hosted lint-format job was retrieved directly: run 39044091242,
job 117429268297. Its [log excerpt](original-lint-format.txt) identifies
tests/test_wizard_render_coverage.py, one file needing reformatting. The prior
README records its main-branch fix at d82092b7 (#2397). That file was not edited
here. Consumer Gate sets format_check: false; a skipped hosted lint-format job
would not prove Black ran.

PR #2498 was verified open and ready (draft=false) on remote head
38433d68f307e5b66d4976ebcfacec54660d9038. That differs from this local measured
source commit; no remote CI result is claimed for the latter. The PR-body update
was rejected because the GitHub tool requires approval and this run's approval
policy is never. The focused acceptance checkbox above is local reconciliation;
no additional full task was marked complete remotely.

Normal git add failed with a read-only .git/index.lock. A separate Git directory
at /tmp/issue-2476-measurement/delivery.git holds the meaningful source commit
above and an evidence-only follow-up, based on the exact local baseline.
/tmp/issue-2476-measurement/issue-2476-measurement.bundle preserves both commits
for a writable checkout. Working-tree changes are also left in place for the
automation system. No push was performed.
