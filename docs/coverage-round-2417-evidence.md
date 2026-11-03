# Coverage round #2417 — closer acceptance evidence

Checked **2026-11-03, America/Chicago**. Tests and behavior from merged
[PR #2419](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2419)
(`982430ef`) are unchanged; this file records the missing execution, mutation,
Gate measurement, and ranking evidence.

## Recovered from PR #2419

- **PR body** (2026-10-20): nine focused tests, no production changes, local
  break/revert claimed but transcripts not archived, new-module-only `--cov`
  table (not Gate-comparable), Gate run cited as authoritative.
- **Merge**: squash `982430ef` on `main`; parent **`80e956ce`** (pre-#2419).
- **Final-head Gate** (PR head `804a2f9`, run
  [39472616859](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/39472616859)):
  **1916 passed, 1 skipped**; product line coverage **14775 statements, 1495
  missing (~90%)** (`line-rate` from `coverage.xml`; `branch = false` in
  `pyproject.toml`).
- **Codex review** on `80c3055`: completed, no findings. CodeRabbit rate-limited.
  No recoverable per-test mutation logs in PR comments.

## Repository Gate invocation

Consumer `.github/workflows/pr-00-gate.yml` → Workflows
`reusable-10-ci-python.yml@main` with `coverage-min: "84"`,
`enable-soft-gate: true`, Python **3.14** (PR #2419 era log shows job label
`python 3.12` on runner **3.14.8**). Local replay used the same pytest/coverage
flags as #2421 closer repair:

```text
.venv/bin/python -m pytest --cov --cov-config=pyproject.toml \
  --cov-report=xml:coverage.xml --cov-report=term-missing \
  --cov-report=json:coverage.json -n 2 --dist=loadgroup
```

Branch diagnostic (non-authoritative; issue wording uses statement+branch):

```text
.venv/bin/python -m pytest --cov --cov-config=pyproject.toml --cov-branch \
  --cov-report=json:coverage-branches.json -n 2 --dist=loadgroup
```

Interpreter: **`.venv/bin/python`** (3.14.3) after `pip install -r requirements-dev.txt -e .`.

## Focused pytest (current `main` + docstring link)

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2417.py
9 passed, 5 warnings in 20.27s
exit=0
```

Revision: `ec9a0127` worktree at closer repair time (includes #2419 tests unchanged).

## Break-and-revert (regenerated 2026-11-03)

Each row: temporary one-line production defect → named node RED → byte-for-byte
restore → GREEN. No mutations committed.

| Test node | Module | Defect (summary) | RED excerpt |
| --- | --- | --- | --- |
| `test_result_explain_tolerates_telemetry_failure_and_missing_metric_delta` | `result_explain.py` | `_with_disclaimer("")` returns `""` | `AssertionError` on disclaimer |
| `test_pa_mapping_helpers_distinguish_mapping_and_top_level_defaults` | `pa.py` | drop mapping `calibration` dict | `AssertionError` on `corr_target` |
| `test_facade_normalizes_scalars_and_applies_all_run_overrides` | `facade.py` | force `return_distribution` `"normal"` | `AssertionError` on `student_t` |
| `test_units_cover_annual_round_trips_and_invalid_units` | `units.py` | drop `sqrt(12)` on monthly→annual vol | `AssertionError` on volatility |
| `test_internal_financing_validates_inputs_and_generates_per_path_draws` | `internal_pa_financing.py` | ignore `n_months <= 0` | `Failed: DID NOT RAISE` |
| `test_manifest_falls_back_when_git_revision_is_unavailable` | `manifest.py` | git fallback `"deadbeef"` not `"unknown"` | `AssertionError` on `git_commit` |
| `test_loader_frequency_and_parameter_error_paths` | `loaders.py` | coerce parse failure to `"monthly"` | `AssertionError` on `"irregular"` |
| `test_sweep_helpers_cover_grid_random_and_invalid_inputs` | `sweep.py` | grid yields `{}` | `AssertionError` on grid combos |
| `test_calibration_report_and_series_id_validation` | `calibration.py` | skip `alpha_ids` length check | `DID NOT RAISE ValueError` |

## Ranking provenance

1. **Dispatch record** — orchestrator log
   `/Users/teacher/.codex/orchestrator/coverage_autopilot.log`:
   `2026-10-20T11:39:07-0500 [Portable-Alpha-Extension-Model#B] opened issue #2417 (lane B, 9 target(s), 180 units)`.
2. **Frozen target order and pre-PR unit counts** — [issue #2417](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2417)
   body (`coverage-autopilot:chunk-round`): escaped-defect priority, then churn,
   then uncovered mass; nine modules totaling **180** statement+branch units at
   dispatch (4+27+23+13+4+2+41+41+25).
3. **Gate configuration** — `[tool.coverage.run] source = ["pa_core", "dashboard",
   "design-system"]`, `branch = false`; authoritative Gate percentage is
   **line-only**; blended branch counts require the diagnostic command above.

## Gate coverage: paired replay (#2419 scope)

Measured on clean git checkouts with the Gate pytest command (local macOS,
xdist `-n 2`; CI Ubuntu — same selection and cov config). Baseline errors
(baseline kit import, unrelated entrypoint failures) match other closer replays;
coverage JSON still emitted.

| Revision | Role | Missing statements (whole product) | Line % |
| --- | --- | ---: | ---: |
| `80e956ce` | Parent of #2419 merge | 1546 | 89.51% |
| `982430ef` | #2419 merge | 1509 | 89.76% |

**Selected nine targets — Gate missing statements**

| Target (issue order) | Before `80e956ce` | After `982430ef` | Δ |
| --- | ---: | ---: | ---: |
| `pa_core/llm/result_explain.py` | 4 | 3 | 1 |
| `pa_core/pa.py` | 27 | 23 | 4 |
| `pa_core/facade.py` | 23 | 19 | 4 |
| `pa_core/units.py` | 13 | 2 | 11 |
| `pa_core/sim/internal_pa_financing.py` | 4 | 1 | 3 |
| `pa_core/manifest.py` | 2 | 0 | 2 |
| `pa_core/data/loaders.py` | 41 | 34 | 7 |
| `pa_core/sweep.py` | 41 | 39 | 2 |
| `pa_core/calibration.py` | 25 | 22 | 3 |
| **Total** | **180** | **143** | **37** |

**Branch diagnostic — combined missing statements + branch exits (same pair)**

| Target | Combined before → after | Reduction |
| --- | ---: | ---: |
| `result_explain.py` | 11 → 8 | 3 |
| `pa.py` | 51 → 43 | 8 |
| `facade.py` | 45 → 37 | 8 |
| `units.py` | 23 → 4 | 19 |
| `internal_pa_financing.py` | 8 → 2 | 6 |
| `manifest.py` | 2 → 0 | 2 |
| `loaders.py` | 74 → 62 | 12 |
| `sweep.py` | 75 → 71 | 4 |
| `calibration.py` | 50 → 40 | 10 |
| **Total** | **339 → 267** | **72** |

### Accepted partial result

- **Gate line reduction on selected targets: 37** (not ~180). **Combined
  diagnostic reduction: 72** (still **108** short of 180).
- PR #2419’s new-module-only `--cov` table measured isolated modules and
  overstated progress versus whole-suite Gate (documented in original PR body).
- Remaining gaps on large modules (`pa.py`, `facade.py`, `loaders.py`,
  `sweep.py`, `calibration.py`) need additional whole-suite tests or production
  paths that are impractical to fake (external I/O, long sweep grids); no claim
  those lines are dead code.
- **No production bug** found during this repair; no threshold or exclusion
  changes.

## Production changes

None (evidence and test docstring link only).
