# Issue #2496 continuation (2026-11-12 UTC)

## Pre-edit acceptance record

Start: 2026-11-12 05:30 UTC. Checked-out revision:
`83cc7ecfa526d2b1a6748f0d35bcdd7c2062061b`; Python 3.14.10 at
`/opt/hostedtoolcache/Python/3.14.10/x64/bin/python`.
Before test edits, `python -m pytest tests/test_coverage_round_2375.py -q
-m "not slow" --disable-warnings` passed: **12 passed, 139 warnings in 4.44s**.
The module still collects all 12 original #2380 cases.

A separate real-input CLI invocation used the module's config/argv helpers,
`data/sp500tr_fred_divyield.csv`, seed 123, one simulation and one month.
It registered `result.xlsx` and `manifest.json`; the workbook's independently
computed and recorded SHA-256 values both were
`af540b06a818fc26fae4e6732e0985b06dbb63d28251a7eef433dceca293fdbd`.
The source manifest recorded `ended_at=2026-11-12T05:30:47.324006+00:00`.
Temporary artifacts were removed after observation.

Acceptance command: `python -m pytest tests/test_coverage_round_2375.py -q
-m "not slow"`, bounded to 120 seconds per subprocess. Desired outcome:
registered completed workbook/manifest references and independent hashes match
the CLI-produced files; both TypeError and ValueError conversion failures reach
the intended converter once and invoke neither animation nor GIF export.
Stop after three unexpected failures at the same step or 75 minutes from start
(06:45 UTC); restore each temporary production mutation in `finally`.

## Live remaining-gap reconciliation

Both requested `gh pr view 2479 --repo stranske/Portable-Alpha-Extension-Model
--json body,files,commits,comments` and `gh pr diff 2479 --repo
stranske/Portable-Alpha-Extension-Model` were attempted again; each exited 1
with `error connecting to api.github.com`. Live authenticated GitHub connector
reads supplied the equivalent PR body, all 14 changed filenames and patches,
comments/reviews, and the five-entry PR commits endpoint. These are live reads,
replacing the cached-page limitation in the original [record](README.md).
The literal `gh` commands remain unavailable; no successful CLI output is claimed.

PR [#2479](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2479)
is **closed, merged=false**, at head
`28b9b47728b819e9c1eaf087890a5b2c1ca59f15`, based on
`7613ffd97a450b15e550f499b7386ec4e67f2c51`.
Its commits endpoint agrees with the local graph: `fd0f7328`, `aac4e80b`,
`51967393`, `70a2016b`, `28b9b477`. The Nov 9 closing comment says the branch
was preserved after its source issue was closed; closure is not merger evidence.
The body still marks final hosted results pending. Review comments independently
flag missing final Gate evidence, unsupported README claims, mutation backup and
interpreter portability, and missing independent config/manifest hashes.

The live `compare_commits(main, 83cc7ecf)` resolves current main to
`ba721af3c4ebad817b58f87bf48bcaa8a2150886`, merge base
`5be77e15c925f5354ab67e2a1749fe6187d2038c`: this branch is two commits ahead
and three behind. Both exact revisions exist locally. Endpoint comparison confirms
no differences in `pa_core/cli.py`, `pa_core/run_artifact_bundle.py`,
`pa_core/llm/compare_runs.py`, or `tests/test_cli_attribution_fallback.py`.
The test module on main still has the weaker original bundle/GIF assertions.
No rebase or unrelated main changes are required for this focused test change.

The following table describes the 05:30 UTC snapshot; the final-status note below
records a merge to main during this run.

| Surface in #2479 | Main at 05:30 / this branch | Remaining work |
| --- | --- | --- |
| `pa_core/cli.py` | Main already has `_np.asarray` through #2492 (`41d8cf11`); branch production is identical. | Retain existing fallback regression and revision inspection; no duplicate production fix. |
| Bundle regression | Main checks only `bundle.json` existence. Branch `83cc7ecf` already checks independent workbook/manifest registration, source bytes/hashes, referenced files, finalized timing, workbook Summary and bundle verification. It also independently checks config/manifest hashes requested in review. | Preserve those checks; keep registration-removal and corrupted-file/hash evidence bound to tested source. |
| GIF conversion regression | Main and #2479 check the message only. Branch already installs reachable exporter/animation mocks and checks zero calls. | Verify both caught exception classes and that the converter is actually reached. |
| Thirteen additional #2479 cases | Daily index, three dashboard failures, two GIF renderer failures, attribution, two trade-off, two registry, two suggestion cases. Main has the separate stronger three-month attribution regression. | Do not import the old branch wholesale; preserve #2380 and assess remaining coverage through matching Gate reports. |
| Twelve historical evidence files | All are confined to `docs/evidence/issue-2375-closer-repair/`, absent from main. #2479 also changes the two Python files above, totaling 14 paths. | Keep those historical results separate from this revision. Do not import the runner with its reviewed backup/interpreter defects. |
| Hosted pytest/CI | #2479 body has no completed final-head result; review confirms this gap. | Obtain actual final-revision links; local results cannot satisfy hosted acceptance. |
| Gate coverage/ranking | Prior ranking names only `compare_runs.py` (rank 1) and `cli.py` (rank 2). Historical baseline run 41696059356 / artifact 13261601545 reports 1/165 missing statements. | Obtain matching baseline/final hosted artifacts and per-target reduction or concrete exceptions. No other target is selected here; focused coverage is not Gate coverage. |

The historical evidence paths are README, acceptance-plan, current-mutation-transcripts,
focused-branches-before/after, gate-baseline, gate-measurements, initial-ci-inspection,
local-validation, mutation-transcripts, ranking-provenance and run_mutations.
Their individual patches were fetched and the two Python patches inspected.
The exact live source identities, commits and changed-path list are retained in
[reconciliation-20261112.json](reconciliation-20261112.json).

The first unchecked task's reconciliation is now substantively available using
equivalent live reads. Exact `gh` transport remains a documented limitation.
The hosted tasks remain incomplete.


## Final candidate verification

The source change parameterizes the existing GIF conversion regression for
ValueError and TypeError and asserts that the intended converter is called once.
The existing zero-animation/zero-export assertions remain. The fake parent viz
module is still installed in both `sys.modules` and `pa_core.viz`; the converter
is patched on the real already-imported `viz_utils` object, and the exporter on
the real `export_backend` object before the CLI's local import. These are reachable
recording mocks, as the production mutations demonstrate.

The bundle case retains all source-byte, file, hash and independent registration
checks. Two local variables (`manifest_ref`, `artifact_bytes`) express the same
assertions in a form both Black and Ruff accept. An optional Ruff format check
initially rejected two assertions already present at `83cc7ecf`; this adjustment
resolved the discrepancy. All evidence was regenerated after that adjustment.
No production files are changed.

The final candidate's `tests/test_coverage_round_2375.py` SHA-256 is
`dfd07637d13a93b03c1a34f535cd136fb6b844aafb1b3727c4c649f6977aceac`. Mutation evidence is bound to this exact content on base `83cc7ecf`,
not to a nonexistent new commit. Every restored production hash matches the
original, and the production diff is empty.

[Final pytest outputs and collection](pytest-20261112.json) record:

- Focused acceptance: **13 passed, 157 warnings in 3.86s**, subprocess wall time
  **6.788s**, under the 120-second bound.
- Related bundle/attribution tests with targeted coverage: **31 passed, 184
  warnings in 35.80s**, subprocess wall time **38.640s**.
- Final collection: **13 cases**; the only additional case is TypeError. The
  original ValueError case now has an explicit parameter ID.

```text
tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts
tests/test_coverage_round_2375.py::test_compare_runs_explains_missing_prior_summary
tests/test_coverage_round_2375.py::test_compare_runs_rejects_missing_prior_manifest
tests/test_coverage_round_2375.py::test_compare_runs_survives_fleet_event_failure
tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]
tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported[value-error]
tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported[type-error]
tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported
tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts
```

[Mutation records](mutations-20261112.json) include exact diffs, named failures,
commands, source/test hashes, restoration hashes and passing reruns for **14**
independent production defects. [Full transcripts](mutation-transcripts-20261112.json.gz)
retain the complete, unabridged JSON; the readable record shortens large binary
assertion explanations and includes raw-output checksums. Each mutation runs the
same full 13-case acceptance module with `-m "not slow" --tb=short
--show-capture=no --disable-warnings`, except the attribution mutation, which
selects the existing dedicated regression. Every mutant exits 1 and every restored
run exits 0; all subprocesses have a 120-second timeout.

| Production mutation | Named failing case | Required diagnostic |
| --- | --- | --- |
| `workbook-registration` | `test_single_run_bundle_contains_the_completed_artifacts` | Missing completed workbook registration |
| `manifest-registration` | `test_single_run_bundle_contains_the_completed_artifacts` | Missing completed manifest registration |
| `manifest-reference` | `test_single_run_bundle_contains_the_completed_artifacts` | Missing completed manifest registration |
| `missing-workbook` | `test_single_run_bundle_contains_the_completed_artifacts` | Missing referenced workbook file |
| `missing-manifest` | `test_single_run_bundle_contains_the_completed_artifacts` | Missing referenced manifest file |
| `workbook-hash` | `test_single_run_bundle_contains_the_completed_artifacts` | Incorrect workbook hash |
| `manifest-hash` | `test_single_run_bundle_contains_the_completed_artifacts` | Incorrect manifest hash |
| `copied-workbook-bytes` | `test_single_run_bundle_contains_the_completed_artifacts` | Bundle workbook differs from CLI-produced file |
| `gif-export-after-conversion-failure` | `test_single_run_gif_conversion_error_is_reported` | GIF export called after conversion failure |
| `gif-animation-after-conversion-failure` | `test_single_run_gif_conversion_error_is_reported` | Animation constructed after conversion failure |
| `attribution-old-numpy-name` | `tests/test_cli_attribution_fallback.py::test_attribution_fallback_exports_annualised_totals` | NameError: name 'np' is not defined |
| `gif-no-type-error-handler` | `test_single_run_gif_conversion_error_is_reported` | TypeError: not numeric |
| `gif-no-value-error-handler` | `test_single_run_gif_conversion_error_is_reported` | ValueError: not numeric |
| `gif-bypass-converter` | `test_single_run_gif_conversion_error_is_reported` | Called 0 times |

The two GIF call-after-failure mutations and converter bypass each fail both
parameter IDs. Narrowing the handler to ValueError fails `[type-error]`;
narrowing it to TypeError fails `[value-error]`. The bundle mutations each fail
the bundle case independently; the other acceptance cases pass. This proves
registration, file, byte and hash sensitivity without relying on `verify()` alone.

The fallback inspection remains `pa_core.cli._main` at base `83cc7ecf`, runtime
NumPy import `_np` at line 417, fallback `_np.asarray(arr).mean()` at line 1555,
with `np` available only under `TYPE_CHECKING` at lines 42–43. The landed fix is
`41d8cf118b124551c1268147fb85b8abbd623542` (#2492). The fresh old-name mutation
reproduces `NameError` in
`tests/test_cli_attribution_fallback.py::test_attribution_fallback_exports_annualised_totals`;
restoration passes. Unmodified current production does not reproduce the defect,
so no duplicate fix or fallback test is added.

## Configured checks and publication limit

[CI inspection](ci-inspection-20261112.json) records the live upstream workflow's
blob identity and commands. [Configured check outputs](configured-checks-20261112.json)
retain both the optional formatter's initial discrepancy and the final passing
Ruff lint, mypy (165 source files), and Ruff format check of the changed module.

[Black outputs](black-20261112.json) record the required repository command:

```text
BLACK_CACHE_DIR=/tmp/issue-2496-black-cache-20261112 black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .
All done! 538 files would be left unchanged.
exit=0
```

The initial attempt failed with a sandbox `PermissionError` while binding Black's
forkserver socket. Using Black's own source discovery and normal single-file CLI
checks sequentially checked all 538 files and populated Black's normal cache;
[the sequential check record](black-sequential-20261112.json) has no failures.
The changed file was checked again in that cache before the exact whole-repository
command passed. No formatting check, rule or test was disabled. Black reports its
normal notebook skip because optional Jupyter dependencies are absent.

The targeted coverage table shows `pa_core/cli.py`: **1107 statements, 473 missed,
57%**; `pa_core/llm/compare_runs.py`: **237 statements, 83 missed, 65%**.
These are focused local measurements, not hosted Gate values. They establish no
120-unit reduction. Ranking provenance and the two historical narrow exceptions
remain as recorded above; they do not waive the entire CLI's uncovered surface.

The pre-continuation branch head has a successful
[Gate run 43160736515](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43160736515)
with coverage artifact `13918275796` (`gate-coverage-3.14-1`, SHA-256
`a3d5ff16aa7aabfe5d6f3c9e1ad07fdc83ea1baaad636b4858635e32ca9d0a81`).
It belongs to `83cc7ecf`, not this candidate. The connector supplied an artifact
file reference, but its ZIP contents were not extracted locally; no counts are
inferred from artifact metadata. This is a potential branch-baseline anchor,
not a matching current-main/final pair or final acceptance evidence.

Local staging failed before changing the index:

```text
git add tests/test_coverage_round_2375.py
fatal: Unable to create '.../.git/index.lock': Read-only file system
```

A direct commit to the existing PR branch was then attempted through the GitHub
connector. Its blob-creation step was rejected by automatic approval review:
`MCP tool call requires approval, but approval policy is never`.
No blob, commit, branch or PR was created. Attempts to add the required
`needs-human` label and blocker comment to PR #2503 received the same rejection.
The blocker is therefore documented locally. No alternative GitHub mutation was
used to evade that rejection.

PR #2503 was initially read as **open, draft=false**, head `83cc7ecf`. The final
read at 05:44 UTC reports **closed, merged=true, draft=false**: an external actor
merged it at `2026-11-12T05:41:09Z`. This run has changed no remote PR state.
Current main is now `828c50970b3c451351016317af6ca40ee288970d`; its test-module
blob is `d64adc8217668b786a8a0f352c0cd3c192ef69c0`, exactly the starting
`83cc7ecf` test blob. Thus main now contains the earlier workbook/manifest
integrity checks and zero-GIF-call checks. This continuation's converter-call
assertion, TypeError case, formatter adjustment and live reconciliation are
still uncommitted. The original comparison table must not be read as a claim
that those earlier assertions are still absent from final current main.

A writable, authorized follow-up must deliver these exact candidate changes and
obtain final-revision hosted pytest, CI and matching coverage artifacts; a merged
PR cannot serve as an open successor. No new branch or PR was created in this
run. The apply-ready patch is saved at
`/tmp/issue-2496-20261112.patch`; it excludes unrelated pre-existing untracked files.

Verified task status (the duplicated issue entries refer to the same work):

- [x] Reconcile #2479 against pinned live current main using equivalent authenticated reads; literal `gh` connectivity remains unavailable.
- [x] Preserve verified completed workbook/manifest registration, existence, bytes and hashes with independent mutation failures.
- [x] Verify zero GIF export/animation calls for both conversion exception types through the intended CLI path.
- [x] Record fallback symbol/revision and the conditional-fix decision with old-defect reproduction.
- [x] Record pre-edit real input, acceptance command, bounded stop, collection and final-content mutation/restoration evidence.
- [ ] Commit the meaningful test changes (blocked by read-only Git and rejected GitHub writes).
- [ ] Obtain final-revision hosted pytest/CI links; working-tree checks alone are insufficient.
- [ ] Complete matching hosted Gate baseline/final comparison and per-target coverage acceptance.
