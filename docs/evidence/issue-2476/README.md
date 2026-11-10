# Issue #2476 evidence — restored coverage round after closed PR #2484

Restores the reviewed #2476 test/evidence commit after closed PR #2484.
The restored focused module is `tests/test_coverage_round_2381.py`
(49 test functions; the 4-case parametrized
`test_results_main_reports_packet_export_failures` brings it to the
recorded **52 cases**). This round adds
`tests/test_coverage_round_2476_facade.py` for the remaining facade
branches below, without changing the restored file's 52-case count.

## Acceptance records (2026-11-10)

Measured code revision: `54219bd739793fa754c3fec319bb2c6b8f8dd898`. The next
commit on `codex/issue-2476` changes only files in `docs/evidence/issue-2476/`.
Interpreter: `.venv/bin/python`, Python 3.14.3 (the CI `python-version`), built
with uv from `requirements.lock` plus `.[dev]`.

```text
python -m pytest -q tests/test_coverage_round_2381.py tests/test_coverage_round_2476_facade.py
63 passed, 18 warnings in 5.00s        (wall 7.64s, under the 120s limit)

python -m black --check --line-length 100 \
  --exclude '(\.venv|\.workflows-lib|node_modules)' .
525 files would be left unchanged.     (exit 0)

pytest --cov --cov-config=pyproject.toml --cov-report=xml:coverage.xml \
  --cov-report=term-missing --cov-report=json:coverage.json -n auto --dist=loadgroup
2 failed, 2032 passed, 1 skipped in 46.05s   TOTAL 14733 stmts, 1169 miss, 92%

(same command) --cov-branch
2 failed, 2032 passed, 1 skipped in 43.82s   TOTAL 14733 stmts, 1168 miss,
4990 branches, 658 partial, 90%
```

Both local failures, `tests/test_dashboard_sample_data.py::test_wheel_install_exposes_bundled_dashboard_samples`
and `tests/test_entrypoints.py::test_console_scripts_work_in_clean_venv`, come from
this host: `ensurepip` dies with SIGABRT inside a nested venv. Neither test touches
the seven targets. Hosted Python 3.14 CI on the PR is the authority for pass/fail.
The 1-statement difference between the two runs is outside the seven targets, which
measured the same in both. Per-file and per-line values are in `gate-coverage.json`
(`snapshots.delivered`).

## Production change

`pa_core/facade.py` `_serialize_agent_semantics_input`: a list mixing pandas
`Series` and `dict` rows now converts each `Series` with `to_dict()` before
`pd.DataFrame(rows)`. On the `origin/main` facade bytes,
`test_serialize_agent_semantics_from_series_and_dict_list` fails with
`AttributeError: 'dict' object has no attribute 'dtype'`. With the fix it passes.
The transcript is in `mutation-transcripts.md`.

## Mutation transcripts

`mutation-transcripts.md` holds 41 production-mutation records covering every added
or materially strengthened test node. Each record gives the red and green exits, the
signals printed, and the sha256 of the production file before mutation, while mutated
and after restoration. The audit of the earlier 39 records is at the top of that file.

## Seven-target round baseline and reduction

The seven ranked targets, in dispatch order (from
`docs/evidence/issue-2381-closer-repair/ranking-provenance.json`), are:

1. `pa_core/llm/result_explain.py`
2. `pa_core/pa.py`
3. `pa_core/facade.py`
4. `pa_core/units.py`
5. `pa_core/sim/internal_pa_financing.py`
6. `pa_core/manifest.py`
7. `dashboard/pages/4_Results.py`

Combined uncovered statements at the round baseline: **209**
(`round_baseline_selected_missing_statements` in
`docs/evidence/issue-2381-closer-repair/gate-measurements.json`).

Reductions from that seven-target round baseline, measured at the revision above:
**207 fewer uncovered statements** (209 → 2) and **119 fewer uncovered branches**
(123 → 4). The branch figures come from the `--cov-branch` supplement and are a
diagnostic only. The authoritative Gate is line-only (`branch = false` under
`[tool.coverage.run]` in `pyproject.toml`).

Chronological attribution (steps between snapshots, not per-file amounts):

| Step | Statements | Branches |
|------|-----------:|---------:|
| round baseline → after merged PR #2384 | 58 | 14 |
| after #2384 → before this work (main up to the #2476 bootstrap) | 131 | 79 |
| before this work → delivered (this PR) | 18 | 26 |

Per-target deltas, round baseline → delivered:

| Target | Statements | Branches |
|--------|-----------:|---------:|
| `pa_core/llm/result_explain.py` | 4 | 7 |
| `pa_core/pa.py` | 27 | 23 |
| `pa_core/facade.py` | 21 | 20 |
| `pa_core/units.py` | 13 | 10 |
| `pa_core/sim/internal_pa_financing.py` | 4 | 4 |
| `pa_core/manifest.py` | 2 | 0 |
| `dashboard/pages/4_Results.py` | 136 | 55 |

## Remaining exceptions (measured)

These are the only seven-target gaps at the measured revision:

- `pa_core/facade.py` statements 750 and 756, arcs 749→750 and 755→756: dead
  post-serialization branches in `export()`. `_serialize_agent_semantics_input` has
  already turned DataFrame and tuple values into lists, so the DataFrame body and the
  tuple-to-list conversion cannot run on the production path. Before the mixed
  Series/dict fix added four lines above them, these were lines 746 and 752.
- `pa_core/pa.py` arc 349→exit: the false exit of `elif args.registry_cmd == "get"`.
  argparse requires a registry subcommand, and only `list` and `get` exist.
- `dashboard/pages/4_Results.py` arc 178→180: the no-message fallthrough in
  `_render_comparison_panel`. Every unavailable `_PreviousRunAvailability` carries a
  message.

Covered at the measured revision, so not exceptions (the earlier README listed
some of these as remaining):

- the numpy `ImportError` fallbacks in `pa_core/facade.py` (lines 50-51, 87-88),
  through import mocking in `test_facade_scalar_helpers_tolerate_missing_numpy`;
- the `except (ValueError, TypeError)` mapping-serialization path, through
  `test_serialize_agent_semantics_keeps_mapping_when_frame_build_fails`;
- the `4_Results.py` `__main__` entrypoint and auto-refresh `st.rerun()` arcs,
  through `runpy` and a fake `st`.

The `TYPE_CHECKING` import block is `# pragma: no cover` and excluded from measurement.

## Lint-format disposition

The historical `lint-format` failure (run 39044091242, `tests/test_wizard_render_coverage.py`)
is retained as recorded in `gate-coverage.json` `lint_format`. That file is outside this
work. The current black check above exits 0.
