# Issue #2403 closer repair — validation report

Repair of merged implementation PR [#2405](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2405). Production code unchanged except transient local mutations during break-and-restore (not committed).

## 1. Focused pytest (required command)

**Interpreter:** `uv run --python 3.14 python` (matches Gate primary runtime).

**Command:**

```bash
cd /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2403__opener
/opt/anaconda3/bin/uv run --python 3.14 python -m pytest \
  tests/test_paths_coverage_2403.py tests/test_schema_export.py tests/test_stress_lab.py
```

**Result (2026-11-02):**

```
.....................                                                    [100%]
21 passed in 1.04s
```

The twelve tests delivered in #2405 are a subset of this module slice (four in `test_paths_coverage_2403.py`, two added in `test_schema_export.py`, six added in `test_stress_lab.py`).

## 2. Break-and-restore evidence (12 tests)

Method: temporarily mutate the covered production path, run the exact pytest node id, record failure, `git checkout --` restore, rerun pass. Full console excerpts: [`mutation-receipts.txt`](mutation-receipts.txt).

| # | Node id | Detects defect? |
|---|---------|-----------------|
| 1 | `tests/test_paths_coverage_2403.py::test_return_distribution_helpers_validate_and_fill_overrides` | yes |
| 2 | `tests/test_paths_coverage_2403.py::test_named_returns_and_sleeve_mapping_cover_mixed_stream_sources` | yes |
| 3 | `tests/test_paths_coverage_2403.py::test_named_returns_rejects_invalid_dimensions_means_and_covariance` | yes |
| 4 | `tests/test_paths_coverage_2403.py::test_named_returns_supports_all_student_t_streams` | yes (assertion strengthened in this repair — see below) |
| 5 | `tests/test_schema_export.py::test_schema_export_supports_all_formats_and_cli_writes_requested_output` | yes |
| 6 | `tests/test_schema_export.py::test_format_default_for_markdown_handles_collections` | yes |
| 7 | `tests/test_stress_lab.py::test_read_index_csv_accepts_getvalue_and_rejects_non_numeric` | yes |
| 8 | `tests/test_stress_lab.py::test_stress_lab_main_runs_sample_flow_and_exposes_all_downloads` | yes |
| 9 | `tests/test_stress_lab.py::test_stress_lab_main_explains_missing_input` | yes |
| 10 | `tests/test_stress_lab.py::test_stress_lab_main_parses_uploaded_csv_and_shows_constraint_deltas` | yes |
| 11 | `tests/test_stress_lab.py::test_stress_lab_main_uses_session_index_when_no_upload` | yes |
| 12 | `tests/test_stress_lab.py::test_stress_lab_main_errors_when_index_resolves_to_none` | yes |

**Assertion repair:** Test 4 originally passed when `draw_named_returns` forced the all-normal fast path while `return_distribution="student_t"`. Added a same-seed comparison of the primary `student_t` draw against a `normal` draw so bypassing the Student‑t branch fails.

## 3. Gate-comparable coverage (parent vs merged)

**Gate test invocation** (from `stranske/Workflows` `reusable-10-ci-python.yml`: `pytest tests/` with `--cov`, reports `coverage.json` / `coverage.xml`; markers from `pyproject.toml` `addopts`: `-m 'not live_llm'`):

```bash
/opt/anaconda3/bin/uv sync --python 3.14 --extra dev
/opt/anaconda3/bin/uv run --python 3.14 python -m pytest tests/ \
  --cov --cov-report=json:coverage.json -m "not live_llm"
```

**Parent revision:** `ce9b37a164302e7b2ebfa41738ea7b76abf9af77` (first parent of #2405 merge `9477281e…`).

**Merged revision:** current `main` / `closer/issue-2403` head including #2405 tests.

**Coverage configuration:** `[tool.coverage.run]` in `pyproject.toml` — `source = ["pa_core", "dashboard", "design-system"]`, `branch = false`. Gate `coverage-min` reads line `line-rate` from `coverage.xml`; branch arms are not measured (`num_branches = 0` in JSON).

### 3.1 Target files (issue #2403 ranking order)

| Target | Parent missed statements | Parent missed branches | Parent combined | Merged missed statements | Merged missed branches | Merged combined | Δ combined |
|--------|-------------------------:|-----------------------:|----------------:|-------------------------:|-----------------------:|----------------:|-----------:|
| `dashboard/pages/6_Stress_Lab.py` | 91 | 0 | 91 | 0 | 0 | 0 | **91** |
| `pa_core/sim/paths.py` | 70 | 0 | 70 | 42 | 0 | 42 | **28** |
| `pa_core/schema.py` | 60 | 0 | 60 | 22 | 0 | 22 | **38** |
| **Totals** | **221** | **0** | **221** | **64** | **0** | **64** | **157** |

**Repository headline (same runs):** `percent_covered` 86.93% → 90.63%.

### 3.2 Shortfall from the ~180 unit chunk goal

Issue #2403 asked for ~180 covered statement+branch units across these targets. Under Gate configuration (lines only), **157** units were removed from the uncovered set (**221 → 64**), **23 short** of 180.

Reasons (not padding / no threshold changes):

- **`paths.py` (28 of 70 ranked units):** Remaining 42 missed statements sit in legacy H/E/M draw paths, mixed copula branches, and orchestration not exercised by named-return helpers alone; covering them would expand scope beyond the #2405 test module.
- **`schema.py` (38 of 60):** Remaining 22 misses are rarely used template / export branches; #2405 already hit CLI, dictionary generation, and markdown default formatting.
- **`6_Stress_Lab.py`:** Ranked gap fully closed on whole-suite measurement (91 → 0).

Branch shortfall is **not applicable** while Gate keeps `branch = false`; combined units equal statement misses.

## 4. Repository ranking evidence

**Recovered (primary):**

- Issue [#2403](https://github.com/stranske/Portable-Alpha-Extension-Model/issues/2403) body (`coverage-autopilot:chunk-round`) lists Orchestrator ranking order: escaped defects → churn → uncovered mass, with explicit targets and pre-PR counts **91 / 70 / 60** (221 total) for Stress Lab, `paths.py`, and `schema.py`.
- Parent Gate-style run at `ce9b37a` reproduces those exact statement misses for the three files (table above), corroborating the issue snapshot.

**Not recoverable locally:**

- The Orchestrator’s internal ranked queue snapshot file at dispatch time (2026-10-19) is not stored in this repository; only the issue text and reproducible parent coverage counts remain as provenance.

**Supporting repo docs (secondary, stale date):** `docs/COVERAGE_GAPS.md` lists `6_Stress_Lab.py` at 27% — useful context but predates #2405 and is not the Gate whole-suite judge.

## 5. CI note

Full-suite local runs reported two environment-specific failures (`test_entrypoints` clean venv wheel probe, `test_dashboard_sample_data` wheel install) unrelated to #2405 tests; coverage JSON was still emitted. Gate on the repair PR is the authoritative CI pytest evidence for the final head.
