# Issue #2476 evidence — restored coverage round after closed PR #2484

Restores the reviewed #2476 test/evidence commit after closed PR #2484.
The restored focused module is `tests/test_coverage_round_2381.py`
(49 test functions; the 4-case parametrized
`test_results_main_reports_packet_export_failures` brings it to the
recorded **52 cases**). This round adds
`tests/test_coverage_round_2476_facade.py` for the remaining facade
branches below, without changing the restored file's 52-case count.

## Current receipts (revision-bound, 2026-11-11)

The [2026-11-12 PNG/Gate repair](png-gate-20261112/README.md) isolates the
external renderer in the CLI PNG test and verifies its figure, path and output.
The complete local Gate-configured statement and branch suites now pass
**2062 cases**, with **3 skipped**, in 65.47s/71.55s using bundled offline
packaging wheels. Three production mutations fail and pass after restoration.
Current ranked gaps remain **2 statements/4 branches**; no additional reduction
is claimed. Matched-policy round-baseline evidence and hosted delivered-head
CI remain open.

The [2026-11-12 statistics assertion verification](statistics-20261112/README.md)
checks exact ranked-target-1 statistics, quantiles and tail rows with six
production mutations. Its identical focused measurements add zero coverage
units. Complete Gate comparison and hosted delivered-head CI remain open.

The [2026-11-12 facade validation](facade-validation-20261112/README.md) checks
duplicate-label rejection and preservation of caller inputs across both row
orders and list/tuple containers. Four production mutations fail the four
cases and pass after restoration. Its identical focused coverage scopes show
zero new units; the full baseline timed out on the PNG CLI test. Hosted
delivered-head CI and matched-policy round-baseline accounting remain open.

The [2026-11-12 helper verification](verification-20261112/README.md) strengthens
the comparison and explanation diagnostics with four production mutations and
records a fresh focused baseline/head diagnostic. It adds zero coverage units;
its full-suite local baseline timed out. The hosted receipts below remain
attributed to their original revisions.

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

Hosted [run 43154194839](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43154194839)
subsequently ran the exact same Gate command and `--cov-branch` supplement via
`.github/workflows/issue-2360-paired-ci-attribution.yml` with runner-local
dependency resolution (sourcing each checkout's `.github/workflows/autofix-versions.env`
and `requirements.lock` on `ubuntu-latest` Python `3.14.10`) on round baseline
`bf424ec6ff58bb88b6431acf7809d42b2c59f95d`, PR `#2384` merge
`1b7f97a1c640ef69efe6a6750cf93f23a105000e`, and prospective delivered merge
`d0e8f8058a0c8f32babf4460390bd2073ed5389a` (merging `6d4d137a` with `main`
`9d083d69`). All six jobs completed on the first attempt within 180 seconds and
recorded their receipts in
[continuation-20261111/hosted-main/seven-targets.json](continuation-20261111/hosted-main/seven-targets.json).

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

Complete statement and branch counts (uncovered / total) across baseline `bf424ec6`, PR `#2384` merge `1b7f97a1`, main `5be77e15`, and delivered merge `d0e8f805` (`ab96b5b4`):

| Target | Baseline `bf424ec6` (stmts / br) | PR `#2384` `1b7f97a1` (stmts / br) | Main `5be77e15` (stmts / br) | Delivered merge `d0e8f805` (stmts / br) |
|--------|---------------------------------:|-----------------------------------:|-----------------------------:|----------------------------------------:|
| `pa_core/llm/result_explain.py` | 4/226 / 7/102 | 4/226 / 7/102 | 0/227 / 4/102 | 0/227 / 0/102 |
| `pa_core/pa.py` | 27/222 / 24/68 | 27/222 / 24/68 | 0/222 / 2/68 | 0/222 / 1/68 |
| `pa_core/facade.py` | 23/332 / 22/120 | 23/332 / 22/120 | 2/332 / 8/120 | 2/337 / 2/124 |
| `pa_core/units.py` | 13/81 / 10/30 | 13/81 / 10/30 | 0/81 / 0/30 | 0/81 / 0/30 |
| `pa_core/sim/internal_pa_financing.py` | 4/35 / 4/18 | 4/35 / 4/18 | 0/35 / 0/18 | 0/35 / 0/18 |
| `pa_core/manifest.py` | 2/61 / 0/4 | 2/61 / 0/4 | 0/61 / 0/4 | 0/61 / 0/4 |
| `dashboard/pages/4_Results.py` | 136/210 / 56/70 | 78/210 / 42/70 | 18/210 / 16/70 | 0/210 / 1/70 |
| **Total** | **209/1,167 / 123/412** | **151/1,167 / 109/412** | **20/1,168 / 30/412** | **2/1,173 / 4/416** |

Per-step statement and branch reductions for all seven ranked targets reconciling the ~180-unit round objective:

| Target | `bf424ec6 → 1b7f97a1` (#2384) stmts / br | `1b7f97a1 → 5be77e15` (intervening) stmts / br | `5be77e15 → d0e8f805` (this PR) stmts / br | Round total (`bf424ec6 → d0e8f805`) stmts / br (units) |
|--------|-----------------------------------------:|-----------------------------------------------:|-------------------------------------------:|-------------------------------------------------------:|
| `pa_core/llm/result_explain.py` | 0 / 0 | −4 / −3 | 0 / −4 | −4 / −7 (−11) |
| `pa_core/pa.py` | 0 / 0 | −27 / −22 | 0 / −1 | −27 / −23 (−50) |
| `pa_core/facade.py` | 0 / 0 | −21 / −14 | 0 / −6 | −21 / −20 (−41) |
| `pa_core/units.py` | 0 / 0 | −13 / −10 | 0 / 0 | −13 / −10 (−23) |
| `pa_core/sim/internal_pa_financing.py` | 0 / 0 | −4 / −4 | 0 / 0 | −4 / −4 (−8) |
| `pa_core/manifest.py` | 0 / 0 | −2 / 0 | 0 / 0 | −2 / 0 (−2) |
| `dashboard/pages/4_Results.py` | −58 / −14 | −60 / −26 | −18 / −15 | −136 / −55 (−191) |
| **Total** | **−58 / −14 (−72)** | **−131 / −79 (−210)** | **−18 / −26 (−44)** | **−207 / −119 (−326)** |

Round attribution:

| Step | Statements | Branches | Source |
|------|-----------:|---------:|--------|
| round baseline → after merged PR #2384 | 58 | 14 | historical test policy |
| after #2384 → before this work (intervening changes) | 131 | 79 | historical test policy |
| main → prospective merge (this PR) | 18 | 26 | current hosted receipts |

Hosted main matches the historical `before_this_work` snapshot exactly, at 20
statements and 30 branches per target, and hosted run 43154194839 with
runner-local dependency resolution independently reproduces the exact
`bf424ec6` (209 statements, 123 branches), `1b7f97a1` (151 statements, 109
branches), and `d0e8f805` (2 statements, 4 branches) counts under `-m 'not slow
and not live_llm'`. The current PR step therefore reconciles with the
historical 18/26. The PR alone removes 44 units; the ~180-unit round objective
is reached across the full round (326 units removed: 72 in #2384, 210 in
intervening changes, and 44 in this PR). The earlier fixed-pin attempt to
measure the round baseline `bf424ec6` with a forced `coverage==7.17.2` CLI pin
failed at dependency installation, before any test ran. Its coverage pin,
7.16.2, conflicts with the matched 7.17.2 (run 43123043013). Under that forced
7.17.2 pin without lock override, the matched round
baseline therefore remains unmeasured; runner-local dependency resolution in
run 43154194839 resolves each checkout's native `.github/workflows/autofix-versions.env`
and `requirements.lock` and confirms all three steps above.

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
