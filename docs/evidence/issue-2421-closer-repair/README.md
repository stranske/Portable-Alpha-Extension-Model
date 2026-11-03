# Issue #2421 closer repair evidence

Checked **2026-11-02, America/Chicago**. This records the missing acceptance
evidence for merged [PR #2422](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2422).
The seven tests and all four production modules retain their existing behavior.
The test module's docstring links this evidence and makes the repair run the
Python Gate instead of receiving only a documentation-path skip.

## Outcome and bounds

Outcome: **7/7 deliberately broken production paths fail their named test, then
pass after byte-for-byte restoration**. The real input is the existing seven-test
module, including its deterministic LLM/telemetry and plugin fakes.

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py
7 passed, 14 warnings in 2.21s
exit=0
```

The stop rule was three failures at the same step or 75 minutes from starting.
Each mutation command has a 110-second subprocess timeout and restores production
in `finally`. Each focused red or green run finished in under two seconds.
No production diff, registry cleanup, coverage exclusions, threshold changes,
additional targets, or new regression tests are part of this repair.

## Recover before regenerating

Read #2422's body, all 22 combined discussion/review entries, final-head statuses,
Gate jobs/logs, and artifacts first. [Recovered evidence](recovered-evidence.json)
records the sources and results. The initial dispatch log retained five named red
failures and a later five-test green run. The receiving lane retained the plugin
red/green transcripts and an independent annual-return mutation. These are useful
historical proofs, but do not supply the requested complete seven-node command,
restoration-hash, and individual-green mapping; the telemetry test also lacked a
mutation. Only that complete protocol and the missing paired coverage measurement
were regenerated. Existing tests were not reimplemented.

The final-head [Gate run 39497784103](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/39497784103)
at `b71c5db824508dc86a1a2a8d9df38a19c4ef5287` succeeded:
**1,928 passed, 1 skipped**, coverage **90.81%**. Its log identifies coverage
artifact `12298178930`, but downloading that artifact now returns **404**.
The retained artifact lists for both final-head Gate runs contain only
`langsmith-fleet.ndjson`, not a coverage payload. The original successful Gate was
reused as CI evidence; its unavailable coverage ZIP was not treated as available.

## Seven-test mapping

Every command is `.venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::<name>`.
Full names, mutations, named failures, actual outputs, and restored SHA-256 values
are in [the concise transcripts](mutation-transcripts.md) and
[machine-readable mapping](mutations.json). The four module hashes agree with
current main and #2422's merge commit; `git diff --exit-code -- pa_core` is clean.

| Test (abbreviated; full names in transcripts) | Temporary defect | Named failure | Restored |
| --- | --- | --- | --- |
| `test_result_helpers_preserve_json_lists_and_prefer_total_metric_row` | Serialize a list as `[]` | `[] != [timestamp, None]` | 1 passed |
| `test_result_explain_tolerates_llm_path_telemetry_failure_and_non_mapping_summary` | Set result status to error when telemetry fails | `'error' != 'success'` | 1 passed |
| `test_units_cover_monthly_to_annual_covariance_and_identity_series` | Remove covariance annualization multiplier | covariance is 1/12 of expected identity | 1 passed |
| `test_internal_financing_uses_default_rng_when_stochastic_rng_is_omitted` | Use seed 123 instead of None | `[(123, 1)] != [(None, 1)]` | 1 passed |
| `test_metrics_reject_invalid_inputs_and_handle_small_samples` | Accept confidence 1 | `DID NOT RAISE ValueError` | 1 passed |
| `test_convergence_diagnostic_rejects_invalid_requests` | Accept checkpoint 0 | `DID NOT RAISE ValueError` | 1 passed |
| `test_metrics_cover_registration_plugins_and_boundary_paths` | Load a plugin without registering it | plugin absent from registry | 1 passed |

These prove discrimination for one relevant defect per test, not mutation
completeness for every assertion. In particular, the telemetry test tolerates an
invoked failing recorder; it does not assert recorder call count. No claim that
deleting the telemetry call is detected is made.

## Gate coverage: actual configuration and paired runs

Before: `543b53597014100c69d12635b428c85ac791aaeb` (**parent of the squash merge**,
not the older PR base `982430ef`).
After: `127f111898a90ee46fb6538f05ede021dd4541d9` (#2422 merge).
Their only tracked difference is the seven-test module.

The consumer `.github/workflows/pr-00-gate.yml` calls the Workflows reusable CI.
The actual historical log resolves it to `d2def1901357855222046823b609a8f87b95ccfb`.
Both revisions configure `source = ["pa_core", "dashboard", "design-system"]`
and **`branch = false`** in `pyproject.toml`.
Therefore authoritative Gate uncovered branches and a blended statement-plus-branch
count are **unmeasured**, not zero. The issue's "blended" label is inconsistent
with this Gate configuration. No measurement-policy change was made to hide this.

Ran the Gate pytest arguments on unmodified Git checkouts in one isolated
environment: Python **3.12.2**, pytest **9.1.1**, coverage **7.16.2**, NumPy
**2.5.1**, pandas **3.0.5**, identical historical `requirements.lock`, and the
original CI's baseline-kit commit `ca04a9c2d096aa905e5913d72a7352b24eb3a113`.
The local replay uses macOS and two xdist workers; original CI used Ubuntu,
Python 3.12.15 and auto workers. Its CI-only collection/resource receipt observer
is omitted locally. Selection, product coverage source/configuration and
`loadgroup` scheduler match. The after result matches the authoritative CI's
14,775 statements, 1,358 missing statements and 90.81% coverage exactly.

```sh
<python> -m pytest --junitxml=pytest-junit.xml --cov --cov-config=pyproject.toml --cov-report=xml:coverage.xml --cov-report=term-missing --cov-report=json:coverage.json -n 2 --dist=loadgroup
```

Before: **1,921 passed, 1 skipped, 953 warnings in 170.85s**, exit 0.
After: **1,928 passed, 1 skipped, 953 warnings in 170.86s**, exit 0.
Whole-product line coverage: **90.57868% -> 90.80880%** (+0.23012 percentage
points), missing statements **1,392 -> 1,358**.
The issue's earlier 89.63% is not this exact-parent baseline and is not reused
as a before measurement. These full Gate replays exceed two minutes; the repair
itself is independently testable with the focused two-second command above.

| Target, original dispatch order | Gate missing statements before | After | Reduction | Constraint/disposition |
| --- | ---: | ---: | ---: | --- |
| `pa_core/llm/result_explain.py` | 3 | 0 | 3 | All Gate-counted statements covered; existing pragma exclusions unchanged. |
| `pa_core/units.py` | 2 | 0 | 2 | All Gate-counted statements covered. |
| `pa_core/sim/internal_pa_financing.py` | 1 | 0 | 1 | All Gate-counted statements covered, including omitted stochastic RNG. |
| `pa_core/sim/metrics.py` | 29 | 1 | 28 | Line 270 is a defensive raise after exhaustive valid-mode returns; invalid modes raise earlier. Do not delete/refactor it or create an impossible-path test. |
| `dashboard/app.py` | 25 | 25 | 0 | Not attempted by #2422; outside this repair's four-target scope. |
| `pa_core/scenario_registry.py` | 25 | 25 | 0 | Not attempted; outside repair scope. |
| `pa_core/sim/sensitivity.py` | 24 | 24 | 0 | Not attempted; outside repair scope. |
| `pa_core/schema.py` | 22 | 22 | 0 | Not attempted; outside repair scope. |
| `pa_core/sim/covariance.py` | 19 | 19 | 0 | Not attempted; outside repair scope. |
| `pa_core/viz/regime_timeline.py` | 14 | 14 | 0 | Not attempted; outside repair scope. |
| `pa_core/llm/tracing.py` | 12 | 12 | 0 | Not attempted; outside repair scope. |
| `pa_core/multi_objective.py` | 12 | 12 | 0 | Not attempted; outside repair scope. |

**Measured Gate reduction: 34 statements; shortfall from approximately 180: 146.**
The first four targets offered only 35 missing Gate statements. #2422 stopped at
four because its local coverage tracer failed, leaving the other eight targets
unattempted (153 statements). Those targets are not asserted to be untestable.
The remaining defensive line makes the selected four-target result 34/35.
This is the explicitly permitted measured partial result, not a claim of 180.

See [Gate transcripts](gate-transcripts.md) and [measurement data](gate-measurements.json).
The failed initial replay environments and their replacements are recorded in
the transcripts; incomplete runs are excluded from the table.

## Supplementary statement-plus-branch diagnostic

To supply the missing branch information without changing Gate, repeated the
same paired suites with **`--cov-branch`** and separate `coverage-branches.json`
outputs. This is a diagnostic measure, **not authoritative Gate branch data**.
Both runs passed: 1,921 before / 1,928 after, each with 1 skipped and 953 warnings
(144.81s / 145.05s). Exact command, counts, missing arcs, and output hashes are in
[branch-diagnostic.json](branch-diagnostic.json).

| Target | Missing statements before -> after | Missing branch exits before -> after | Combined uncovered before -> after | Reduction |
| --- | ---: | ---: | ---: | ---: |
| `pa_core/llm/result_explain.py` | 3 -> 0 | 5 -> 4 | 8 -> 4 | 4 |
| `pa_core/units.py` | 2 -> 0 | 2 -> 0 | 4 -> 0 | 4 |
| `pa_core/sim/internal_pa_financing.py` | 1 -> 0 | 1 -> 0 | 2 -> 0 | 2 |
| `pa_core/sim/metrics.py` | 29 -> 1 | 30 -> 2 | 59 -> 3 | 56 |
| **Selected total** | **35 -> 1** | **38 -> 6** | **73 -> 7** | **66** |

The diagnostic combined reduction is **66**, a **114-unit shortfall from 180**.
Whole-product diagnostic coverage is 87.90286% -> 88.23678%; its denominator is
14,775 statements + 4,990 branch exits. It cannot be compared directly to the
line-only Gate percentage. Partial-branch counts in the terminal report are not
missing-branch counts; the table uses JSON `missing_branches`.

The four remaining result-explanation exits are skipping an invalid/empty secret,
falling through an empty response-content list, ignoring a non-string/non-mapping
content item, and ignoring a non-mapping delta item. They were not exercised by
these seven existing tests, not established to be untestable. Metrics retains the
unreachable defensive mode exit (268 -> 270) and the valid but unexercised path
that suppresses an extra metric definition colliding with a built-in name
(888 -> 887). Adding tests for those paths would expand the requested repair.
Units and internal financing have no remaining missing statements or branch exits
in this supplementary measure.

## Ranking provenance

[Issue #2421](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2421)
is the retained ordered dispatch record: escaped-defect priority, then churn,
then uncovered mass; the selected prefix was result explanation (churn 3),
units (churn 2), internal financing (churn 2), then metrics (29 missing units).
The dispatch log records lane B, round 3, 12 targets and 188 units at
2026-10-20T18:41:37Z. This repair preserves that selection instead of reranking it.

[Ranking provenance](ranking-provenance.json) also preserves the relevant rows
and SHA-256 of the later 2026-10-20 target cache. It corroborates the churn tiers
and zero tier-1 signals in that later snapshot, but was refreshed after the first
test commit: its raw positions and gaps cannot establish the exact original
pre-dispatch ranking. That original raw JSON/tier-1 input was not recovered.
Older repository ranking records for #2336 and the 2025 coverage-gap document
were inspected and are not substituted for the #2421 dispatch record.
