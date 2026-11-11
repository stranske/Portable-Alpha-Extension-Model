# Issue #2476 evidence — restored coverage round after closed PR #2484

Restores the reviewed #2476 test/evidence commit after closed PR #2484.
The restored focused module is `tests/test_coverage_round_2381.py`
(49 test functions; the 4-case parametrized
`test_results_main_reports_packet_export_failures` brings it to the
recorded **52 cases**). This round adds
`tests/test_coverage_round_2476_facade.py` for the remaining facade
branches below, without changing the restored file's 52-case count.

## Current receipts (revision-bound, 2026-11-11)

These hosted receipts measure the delivered code. Sections headed
**historical snapshot** below record earlier revisions under the earlier test
policy. They are kept for attribution and do not describe the current head.

Hosted [run 43122605282](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43122605282)
ran the Gate command
`python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml:coverage.xml --cov-report=term-missing --cov-report=json:coverage.json -n auto --dist=loadgroup -m 'not slow and not live_llm'`
and, separately, the same command with `--cov-branch`. It measured current
main `5be77e15c925f5354ab67e2a1749fe6187d2038c` and the prospective merge
`ab96b5b495547fcad317da7c135d9b082aa0e150`, which is candidate `4b1f5e4c`
merged with that main. Each of the four jobs passed on one attempt within
180 seconds. Receipts are in
[continuation-20261111/hosted-main/](continuation-20261111/hosted-main/seven-targets.json).
Details are in the [continuation evidence](continuation-20261111/README.md).

Per-target deltas, main → prospective merge:

| Target | Statements | Branches |
|--------|-----------:|---------:|
| `pa_core/llm/result_explain.py` | 0 | −4 |
| `pa_core/pa.py` | 0 | −1 |
| `pa_core/facade.py` | 0 | −6 |
| `pa_core/units.py` | 0 | 0 |
| `pa_core/sim/internal_pa_financing.py` | 0 | 0 |
| `pa_core/manifest.py` | 0 | 0 |
| `dashboard/pages/4_Results.py` | −18 | −15 |
| **Total (this PR)** | **−18 (20 → 2)** | **−26 (30 → 4)** |

Round attribution:

| Step | Statements | Branches | Source |
|------|-----------:|---------:|--------|
| round baseline → after merged PR #2384 | 58 | 14 | historical test policy |
| after #2384 → before this work (intervening changes) | 131 | 79 | historical test policy |
| main → prospective merge (this PR) | 18 | 26 | current hosted receipts |

Hosted main matches the historical `before_this_work` snapshot exactly, at 20
statements and 30 branches per target. The current PR step therefore
reconciles with the historical 18/26. The round baseline (209 statements,
123 branches), the #2384 step and the intervening step still come from the
historical test policy. The PR alone removes 44 units. The ~180-unit round
objective is reached only when those historical steps are counted. The one
matched-policy attempt to measure the round baseline `bf424ec6` failed at
dependency installation, before any test ran. Its coverage pin, 7.16.2,
conflicts with the matched 7.17.2 (run 43123043013). The matched round
baseline therefore remains unmeasured.

Residual exceptions at the prospective merge: `pa_core/facade.py` statements
751 and 757 with arcs 750→751 and 756→757, `pa_core/pa.py` arc 349→exit, and
`dashboard/pages/4_Results.py` arc 178→180. The historical sections give the
same gaps as facade lines 750 and 756, from before the four-line fix.

## Acceptance records (2026-11-10) — historical snapshot

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

## Seven-target round baseline and reduction — historical snapshot

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

## Remaining exceptions (measured at `54219bd7`) — historical snapshot

These were the only seven-target gaps at the measured revision:

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

Run [39044091242](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/39044091242)
failed at `black --check --line-length 100 --exclude '(\.venv|\.workflows-lib|node_modules)' .`:
`would reformat tests/test_wizard_render_coverage.py` (1 file would be reformatted,
504 unchanged). Main fixed that file in `d82092b74c248cd73a8834e2b67d7d78f3754b6f`
(PR #2397, 2026-10-18). Its current bytes match main; this continuation did not edit it.

Independent check on 2026-11-10 at `24b5f415e2399322b5e8c03359edf8c3ae9b3cf5`,
using `.venv/bin/python` (3.14.3), ran the exact Black command above with `-m black`:
exit 0, **525 files would be left unchanged**.

PR Gate run [42905705483](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/42905705483)
on that exact head reports pytest **SUCCESS**, `2034 passed, 1 skipped, 1014 warnings
in 146.32s`, plus successful Ruff, mypy and gate-summary. The two local nested-venv
ensurepip failures are absent in hosted CI; neither test was changed. Its lint-format
job is **SKIPPED**, because `pr-00-gate.yml` explicitly sets `format_check: false`.
A skipped job is not evidence that Black ran. The reusable lint-format job runs Black;
Ruff and mypy have separate jobs. No workflow or dependency pin was changed.

## Independent continuation spot-checks (2026-11-10)

At `24b5f415e2399322b5e8c03359edf8c3ae9b3cf5`:

- Focused command above: **63 passed, 18 warnings in 5.69s**, under two minutes.
- Three transcript mutations reproduced in **34.23 seconds total**: redaction token,
  explain-panel workbook forwarding, and mixed Series/dict facade serialization.
  Each named test exited 1 with the recorded failure, then exited 0 after restoration.
  All three mutated and restored sha256 values matched their transcript records.
  The facade mutation reproduced `AttributeError: 'dict' object has no attribute 'dtype'`.
- The headline Results test checks metric labels and values (2.00%, -1.00%, 3.00%,
  4.00%), summary identity, manifest/workbook forwarding, and PNG/async export
  arguments. Dedicated explain and sweep-export tests assert delegated arguments.
  Some older visualization smoke tests check only output presence; they were left
  unchanged and are not relied on for argument correctness.

This continuation's final commit changes only `docs/evidence/issue-2476/`.
Production and test bytes remain those of the checked head above. The original
coverage measurements remain attributed to `54219bd7`; coverage was not remeasured.

Later commits added tests and strengthened existing cases, so these historical
measurements do not establish coverage for the delivered code and test revision.
The [calibration CLI continuation](calibration-20261111/README.md) adds seven
ranked-target-2 cases and records twelve production mutations. Its matched
focused measurements cover 15 statements and 4 branches; it does not establish
a fresh whole-suite Gate reduction or completion of the round.
The scoped [2026-11-11 review](review-20261111.md) records a revision-bound
focused branch supplement, which cannot substitute for the full Gate
measurement. The hosted whole-suite receipts in **Current receipts** above
superseded the request for a fresh full Gate run.


Explicit CI run [42924499448](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/42924499448)
on `24b5f415e2399322b5e8c03359edf8c3ae9b3cf5` ran **lint-format: SUCCESS**
(job `130207525801`), **lint-ruff: SUCCESS**, and **typecheck-mypy: SUCCESS**.
Its pytest job was still running at this evidence commit; the completed pytest
result for the same head is run `42905705483` above. This dispatch exercises
Black without changing the PR Gate policy. Fresh CI on the final evidence-only
head is checked separately in the continuation result.
