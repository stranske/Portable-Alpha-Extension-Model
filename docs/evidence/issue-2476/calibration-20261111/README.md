# Ranked target 2: calibration CLI continuation

Checked 2026-11-11 UTC with the repository-selected Python 3.14.10.
Baseline: `496cf6c7884ada74bc1e0586075af28d91a3d486`.
Measured source commit: `7e3e46965c3713d1acdb95cca59f704b6b5c3dbc`.
Available main remains `5be77e15c925f5354ab67e2a1749fe6187d2038c`.
The later evidence commit leaves the measured Python bytes unchanged.

The first unchecked measurement task remains blocked by the previously recorded
180-second whole-suite timeout in
[measurement-20261111/contract.json](../measurement-20261111/contract.json).
Per the attempt/time stop rule, that run was not repeated. This continuation
advances the next task by adding behavioral tests for ranked target 2; it does
not claim a completed seven-target task or fresh whole-suite Gate reduction.

Before editing tests, re-read the issue-format and target-environment guides,
the previous candidate review, measurement contract and tests. Repeated the
requested `rg -n 'coverage|pytest|lint-format|Gate' .github pyproject.toml`
inspection, then confined follow-up searches to workflow/configuration files
to avoid the existing untracked node_modules. The consumer workflows delegate
to reusable Python CI. The recorded Gate command is:

```sh
python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml:coverage.xml --cov-report=term-missing --cov-report=json:coverage.json -n auto --dist=loadgroup -m 'not slow and not live_llm'
```

The unchanged product coverage configuration has branch=false. The comparable
focused runs below use the same configuration and add `--cov-branch` for the
requested branch diagnostic. They use the same five test modules, adding seven
cases within the existing focused module. Only the report destination differs
between the commands recorded in [baseline.json](baseline.json) and
[head.json](head.json). Both extracts retain exact revisions, configuration,
test and production hashes, coverage metadata, totals, missing lines and arcs.
[contract.json](contract.json) was recorded before the test edits.

Stop rule: reuse the documented full-suite timeout; one baseline and one head
focused run, each capped at 120 seconds; at most three attempts of a failing
step. Do not advance past a ranked target without measured saturation or a
documented exception. The baseline passed 134 cases in 11.68 seconds; head
passed 141 in 11.69 seconds (13.736 seconds wall).

| Rank / target | Statements | Missing before → after | Branches | Missing before → after |
| --- | ---: | ---: | ---: | ---: |
| 1. pa_core/llm/result_explain.py | 227 | 0 → 0 | 102 | 0 → 0 |
| 2. pa_core/pa.py | 222 | 34 → 19 | 68 | 14 → 10 |
| 3. pa_core/facade.py | 333 | 15 → 15 | 120 | 14 → 14 |
| 4. pa_core/units.py | 81 | 3 → 3 | 30 | 3 → 3 |
| 5. pa_core/sim/internal_pa_financing.py | 35 | 0 → 0 | 18 | 0 → 0 |
| 6. pa_core/manifest.py | 61 | 2 → 2 | 4 | 2 → 2 |
| 7. dashboard/pages/4_Results.py | 210 | 17 → 17 | 70 | 7 → 7 |

The recorded issue #2381 order is unchanged. Target 1 is saturated in this
scope, permitting work on target 2. Targets 3–7 were measured, not advanced.
Target 7's historical 136 uncovered units is not a current measurement.
The historical full-suite saturation claims for pa.py (one parser exit arc)
and facade.py (two defensive post-serialization lines/arcs) remain attributed
to their historical revisions. This focused suite establishes neither claim.
The reason #2384 originally selected target 7 first remains unknown.

This continuation covers **15 statements and 4 branches**, all in pa.py; see
[deltas.json](deltas.json). No production file changed. Remaining target 2 gaps
include CSV conversion fallbacks, top-level forwarding, sweep metadata and
scenario output. Target 2 and the overall ranked task remain incomplete.

Historical records separately attribute 58 statements/14 branches to #2384,
131/79 to intervening work, and 18/26 to the earlier #2476 implementation:
326 units against the approximate 180-unit round objective. This continuation's
19 focused units cannot be added to that whole-suite total: some of these paths
already have integration coverage outside the focused scope. These results
prove a focused reduction, not another 19 units of whole-suite round reduction.
Fresh comparable whole-suite head evidence is still required for acceptance.

The seven new cases verify default, flat/nested template and CLI settings;
the template's observation threshold and short-sample flag; input frame and
index forwarding; YAML export result/destination; and rejection of invalid
settings before any data load or calibration. Existing calibration integration
tests separately exercise the real importer and calibrator implementations.
The new collaborator mocks are not evidence for their implementation coverage.

All seven new cases have actual production mutations with named failures and
restored passing reruns. Twelve checks additionally verify frame forwarding,
export destination, template observation thresholds and rejection before I/O.
See [complete transcripts](mutation-transcripts.md) and
[commands, exact mutations, exit codes, timings and hashes](mutations.json).
The `calibration-frame` mutation (`df` → `df.copy()`) is **identity-only**. The copy
has equal contents, and only the test's `is frame` identity assertion detects it.
It shows that the parsed frame object is forwarded. It does not show that any
frame content is checked.
Production was restored in finally after each red run; hashes for all seven
target files match the baseline. No demonstrated production fix was needed.

- [x] Focused acceptance: `python -m pytest tests/test_coverage_round_2381.py -m 'not slow and not live_llm' -q` passed 59 cases in 4.18 seconds (7.383 seconds wall), below two minutes. [Receipt](focused-acceptance.json).
- [x] Matched focused coverage: 141 cases pass; measured statement/branch deltas above.
- [x] All twelve mutations fail their named tests and pass after byte restoration.
- [x] Related evidence, previous-run, validation and calibration integration tests: 32 pass in 2.99 seconds.
- [x] Ruff repository command passes.
- [x] Black formatted the modified test at line length 100; repository check passes with 538 Python files unchanged using the existing sandbox launcher.
- [ ] Fresh whole-suite Gate baseline/head comparison and required hosted CI on this delivered source commit.

Captured commands and full output are in [run-output.txt.gz](run-output.txt.gz)
(`gzip -dc docs/evidence/issue-2476/calibration-20261111/run-output.txt.gz`).
The initial 63-case candidate run preceded the explicit parameter ID edit;
all subsequent acceptance, mutation and head coverage runs use the final IDs.
The baseline/head coverage command scope and configuration are identical.

The direct required Black command failed because Python 3.14's forkserver
cannot bind a Unix socket in this sandbox. The existing
[sandbox launcher](../measurement-20261111/black-sandbox-launch.py.txt) runs the
same installed Black, discovery, formatter and flags using fork plus an asyncio
completion poll. Its passing command is captured with the direct failure.
The original hosted lint-format failure was
[test_wizard_render_coverage.py](../measurement-20261111/original-lint-format.txt),
fixed previously on main at d82092b7 (#2397); this continuation does not edit it.

Read-only GitHub checks confirm PR #2498 is open with draft=false and remote
head equal to the local baseline. Gate run 43102502499 on that baseline has
successful pytest, Ruff, mypy and gate-summary jobs; lint-format is skipped.
Those results do not establish hosted CI for the new source commit. Coverage
artifact 13892058074 is listed for that run, but its downloaded file reference
could not be transferred into the sandbox (URLError); no artifact contents or
whole-suite target measurements are inferred from the artifact listing.
No remote PR state or checkboxes were changed.

Normal git add fails because .git/index.lock is on a read-only filesystem.
The meaningful source commit and evidence follow-up are preserved in
`/tmp/issue-2476-calibration/delivery.git`, based on the exact baseline above.
`/tmp/issue-2476-calibration/issue-2476-calibration.bundle` preserves both commits
for a writable checkout; changes are also left in the working tree for
automation. No push was performed. Workflow and other protected files are
unchanged. Only the locally verified acceptance checkbox is checked above;
the broader outstanding tasks remain open.
