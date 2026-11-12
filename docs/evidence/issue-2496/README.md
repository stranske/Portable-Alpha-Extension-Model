# Issue #2496 acceptance evidence

Continuation: [2026-11-12 live reconciliation and validation](continuation-20261112.md)
supersedes the live-data limitation below using authenticated GitHub connector
reads. The sections below preserve the original 2026-11-11 record.

## Before implementation (2026-11-11 UTC)

Starting revision: `0e56868bcf7439ef93a3b18e7a73434bf770a12a`.
Local `origin/main`: `5be77e15c925f5354ab67e2a1749fe6187d2038c`.
Python: `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python` (3.14.10).

The unmodified acceptance module ran with
`python -m pytest tests/test_coverage_round_2375.py -q -m "not slow"`:
**12 passed, 139 warnings in 3.74s**. A separate invocation of the same CLI
helpers with the repository's `data/sp500tr_fred_divyield.csv`, seed 123,
one simulation and one month produced a workbook and a completed bundle.
Its metadata registered `result.xlsx` as `outputs/result.xlsx`, and
`manifest_path` as `manifest.json`. The source workbook SHA-256 and recorded
output hash both equalled
`8ee8776da96c92ae10d6a88081c1df5ce3b9adf6c4d6f8f4237c056cb7bb1aae`.
The source manifest had `ended_at=2026-11-11T19:38:01.983863+00:00`.
This observation establishes the real-input baseline; the old test asserted
only that `bundle.json` existed. An initial standalone harness attempt resolved
its relative `__file__` after changing directory and failed to find the index;
resolving the test path before changing directory corrected that harness error.

Acceptance command: `python -m pytest tests/test_coverage_round_2375.py -q
-m "not slow"`, with a 120-second subprocess timeout. Intended outcome:
the CLI-produced completed workbook and manifest are registered, referenced
files exist, and independent hashes match their source files. Conversion
failure must call neither animation construction nor GIF export. Each changed
case must reject a production mutation and pass after restoration.
Stop after three unexpected failures at one step or 75 minutes from
19:38 UTC; stop individual pytest runs at 120 seconds. Hosted results remain
pending until they are available for the final revision.

## Remaining-gap reconciliation

Both requested commands were attempted before edits:

```text
gh pr view 2479 --repo stranske/Portable-Alpha-Extension-Model --json body,files,commits,comments
gh pr diff 2479 --repo stranske/Portable-Alpha-Extension-Model
```

Both exited 1: `error connecting to api.github.com`. Thus the exact live JSON,
full comments and diff reconciliation remains blocked. No current-main freshness
claim is made beyond the local ref above. The public
[PR #2479 conversation](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2479)
was readable through the browsing tool (cached last week); it exposes the body,
five commits and review comments, but is not an authoritative live status check.
The local PR branch resolves to the same head named in that body:
`origin/closer-r3/issue-2375@28b9b47728b819e9c1eaf087890a5b2c1ca59f15`.
Inspected locally with:

```text
git log --oneline origin/main..origin/closer-r3/issue-2375
git diff --name-status origin/main...origin/closer-r3/issue-2375
git diff origin/main...origin/closer-r3/issue-2375 -- pa_core/cli.py tests/test_coverage_round_2375.py
git diff origin/main origin/closer-r3/issue-2375 -- pa_core/cli.py
```

The five commits are `fd0f7328`, `aac4e80b`, `51967393`, `70a2016b`,
and `28b9b477`. Its merge-base diff has 14 files: the CLI, the test module,
and 12 files under `docs/evidence/issue-2375-closer-repair/` (README,
acceptance plan, two mutation transcripts, runner, two focused branch reports,
Gate baseline, Gate measurements, CI inspection, local validation and ranking
provenance). Those evidence files are absent from current main.

| Area | Prior repair versus current main | Remaining work |
| --- | --- | --- |
| Bundle integrity | PR #2479 requires the workbook, checks bytes/hashes and loads/verifies the bundle; main still checks only metadata-file existence. The prior repair does not compare the registered manifest with the finalized CLI manifest or independently check its hash. | Strengthen the existing test for workbook **and** manifest registration, existence, source contents and hashes; prove each failure independently. |
| GIF conversion | Both versions patch real `viz_utils.safe_to_numpy` after replacing `pa_core.viz`; both assert only the conversion message. | Install reachable recording animation/export doubles and assert zero calls after the CLI conversion failure. |
| Attribution | PR #2479 changes `np.asarray` to `_np.asarray`. Main already contains that production change via `41d8cf11` (#2492), plus a stronger three-month regression in `tests/test_cli_attribution_fallback.py`. Endpoint diff of the CLI is empty. | Record the actual symbol/revision and reproduce the old defect by mutation; do not duplicate the production fix. |
| Additional cases | PR #2479 adds 13 cases (25 total): daily index, three dashboard launch failures, two GIF renderer failures, attribution, two trade-off, two registry and two suggestion cases. Main retains the 12 original cases. | These are not imported wholesale in this focused integrity repair. Existing #2380 cases must remain collected and passing. |
| Mutation evidence | The old branch has 25 selected mutation failures/restorations, bound to its old head. Workbook-registration removal is covered; independent manifest-registration removal is not. | Capture fresh mutation evidence for every materially strengthened case on this branch. Historical evidence does not validate this revision. |
| Hosted checks | The visible PR body says final-head validation is pending at `28b9b477`; the visible comments are review activity, not final pytest/Gate results. | Obtain final-revision hosted pytest/CI links and matching baseline/final Gate artifacts. No hosted result is inferred from old checked boxes. |

Ranking provenance in the prior branch names issue #2375, rank 1
`pa_core/llm/compare_runs.py`, rank 2 `pa_core/cli.py`; no new production
coverage target is selected here. Its Gate baseline names run `41696059356`,
artifact `13261601545`, Python 3.14.9, coverage 7.17.0, branch=false, product
sources `pa_core`, `dashboard`, `design-system`, and two xdist workers. It reports
1 and 165 missing statements respectively. Its final measurements are pending.
The focused branch probes (CLI 556 to 449 missing statements) and historical
Python 3.12 counts (11/211) cannot stand in for a matching final Gate report.
The prior exceptions name an internal sentinel-only formatter path in
`compare_runs.py` and an unreachable sweep sensitivity branch in the CLI;
those are historical inspection claims requiring revalidation before acceptance.

The first task stays unchecked because the required live `gh` outputs are
unavailable. Work proceeds to the next task under the stated blocked-task rule.

## Verified implementation and collection

The strengthened tests keep all 12 existing collected cases; no case was removed
or added. Collection with `python -m pytest tests/test_coverage_round_2375.py
--collect-only -q -m "not slow"` enumerated:

1. `test_compare_run_helpers_reject_unusable_artifacts`
2. `test_compare_runs_explains_missing_prior_summary`
3. `test_compare_runs_rejects_missing_prior_manifest`
4. `test_compare_runs_survives_fleet_event_failure`
5. `test_metric_catalog_rejects_non_dataframe_and_empty_delta`
6. `test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]`
7. `test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]`
8. `test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]`
9. `test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]`
10. `test_single_run_gif_conversion_error_is_reported`
11. `test_single_run_dashboard_missing_file_is_reported`
12. `test_single_run_bundle_contains_the_completed_artifacts`

Only cases 10 and 12 changed. The bundle case independently checks the workbook
in `outputs` and the manifest in `manifest_path`, then config/manifest/workbook
file existence, source bytes and SHA-256 hashes, all other output references,
finalized manifest timing, the manifest's workbook hash, a nonempty Summary
sheet, and `RunArtifactBundle.load(...).verify()`.

Inspection of `_install_fake_viz` confirmed that it replaces both
`sys.modules["pa_core.viz"]` and `pa_core.viz`. The previously imported real
`viz_utils` remains the module used by the CLI's local `safe_to_numpy` import.
The test patches that object directly. It adds `animation.make` to the fake
parent module and patches the real `export_backend.write_figure_image` object
before the CLI's local import. The doubles are reachable, recording mocks:
mutations below prove both call-count assertions can fail. The conversion-error
message and completed workbook assertions still pass through the same CLI route
selected by `--sensitivity`.

The manifest registration mutation changes the CLI's `RunArtifact` construction
from `manifest=manifest_data` to `manifest=None`. The separate reference mutation
keeps the manifest file but removes its metadata registration. Merely deleting
`_record_artifact(manifest_json)` would not remove this bundle registration:
finalization records it again, and `_build_outputs_map` deliberately excludes
the source manifest from `outputs`. The tests follow the actual bundle schema.

## Production mutation and restoration evidence

[mutations.json](mutations.json) records the base revision, exact production and
test SHA-256 values, individual source diffs, subprocess commands, named failure
excerpts, elapsed times, restoration hashes and passing reruns. These are fresh
working-tree experiments based on `0e56868b`, bound to the tested file contents;
they are not executions on the old PR head. The strengthened module's SHA-256 is
`8b63887695585a58c5d6dd18063e7be2cec3b0c4d6fe662ab4da8c616b99b64a`.
Every production restoration matches its original hash, and the final production
diff is empty. Raw subprocess-output hashes accompany excerpts; the enormous
binary workbook assertion explanation and captured CLI logs are abbreviated.

All first ten mutations run the entire 12-case acceptance module with
`-q -m "not slow" --tb=short --disable-warnings`, under a 120-second timeout.
Each exits 1 with exactly the named failing case below and 11 passing cases;
after restoration the identical command exits 0 with 12 passing cases.
The attribution mutation selects the existing regression by full node ID and
likewise exits 1, then 0 with one passing case. One preliminary runner attempt
stopped at its unique-source-match guard before the manifest mutation; the
selector was narrowed and the full matrix was rerun. The runner restored all
production files in `finally` on that attempt as well.

| Production mutation | Named failing case | Required diagnostic |
| --- | --- | --- |
| Remove CLI workbook registration | `test_single_run_bundle_contains_the_completed_artifacts` | Missing completed workbook registration |
| Remove CLI manifest payload registration | `test_single_run_bundle_contains_the_completed_artifacts` | Missing completed manifest registration |
| Remove manifest metadata reference | `test_single_run_bundle_contains_the_completed_artifacts` | Missing completed manifest registration |
| Delete copied workbook | `test_single_run_bundle_contains_the_completed_artifacts` | Missing referenced workbook file |
| Delete copied manifest | `test_single_run_bundle_contains_the_completed_artifacts` | Missing referenced manifest file |
| Record incorrect workbook hash | `test_single_run_bundle_contains_the_completed_artifacts` | Incorrect workbook hash |
| Record incorrect manifest hash | `test_single_run_bundle_contains_the_completed_artifacts` | Incorrect manifest hash |
| Corrupt copied workbook and rehash it | `test_single_run_bundle_contains_the_completed_artifacts` | Bundle workbook differs from CLI-produced file |
| Invoke exporter in conversion exception handler | `test_single_run_gif_conversion_error_is_reported` | GIF export called after conversion failure |
| Continue from conversion exception into animation | `test_single_run_gif_conversion_error_is_reported` | Animation constructed after conversion failure |
| Restore old `np.asarray` name | `test_attribution_fallback_exports_annualised_totals` | NameError: name 'np' is not defined |

## Attribution symbol and revision inspection

At `0e56868b`, the fallback is inside `pa_core.cli._main`, lines 1549–1566.
The module's `np` import is under `TYPE_CHECKING` (lines 42–43); `_main` imports
NumPy as `_np` at line 417 and the fallback uses `_np.asarray(arr).mean()` at
line 1555. The relevant landed fix is
`41d8cf118b124551c1268147fb85b8abbd623542` (#2492). It also added the existing
three-month `tests/test_cli_attribution_fallback.py` regression: a forced
`TypeError` from `compute_sleeve_return_attribution` reaches this fallback through
`pa_core.cli.main` and checks unique Total rows for every workbook agent and
annualized means against raw return sheets. It passes on the starting revision.
Changing only `_np` back to `np` reproduces the old NameError in that named case;
restoring it passes. The defect does not reproduce on current unmutated main,
so this repair adds no duplicate fallback test or production fix.

## Local validation and remaining hosted work

The repository's `ci.yml` selects Python 3.14, lint, type checking and coverage
through the [upstream reusable workflow](https://github.com/stranske/Workflows/blob/main/.github/workflows/reusable-10-ci-python.yml).
Its inspected Ruff and mypy commands were run locally:

```text
python -m ruff check --output-format github --extend-exclude .workflows-lib .
exit=0
python -m mypy --config-file pyproject.toml --exclude .workflows-lib .
Success: no issues found in 165 source files
exit=0
python -m black --line-length 100 tests/test_coverage_round_2375.py
1 file reformatted
exit=0
```

The required whole-repository Black command first failed because the sandbox
denied the multiprocessing forkserver socket (`PermissionError`). It was rerun
with `BLACK_NUM_WORKERS=1`, preserving the requested paths, exclusions and checks.
Notebook dependencies are absent, so Black reports its normal notebook skip;
the changed Python file is included.
The single-worker run stalled; bounded 20/60-second diagnostic runs exited 124.
After exhausting these worker retries, verification switched to sequential
execution: all 538 discovered files were checked individually through Black's
normal CLI entry point, with the repository configuration and a writable
`BLACK_CACHE_DIR=/tmp/issue-2496-black-cache`. Black itself populated the cache;
no formatting rules or validation checks were disabled. The exact required
whole-repository command was then rerun using that cache.

```text
BLACK_CACHE_DIR=/tmp/issue-2496-black-cache black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .
All done! 538 files would be left unchanged.
exit=0
```

Related targeted tests and coverage (a local probe, **not** a hosted Gate report):

```text
python -m pytest tests/test_coverage_round_2375.py tests/test_cli_attribution_fallback.py tests/test_run_artifact_bundle.py -q -m "not slow" --cov=pa_core.cli --cov=pa_core.llm.compare_runs --cov-report=term-missing
30 passed, 166 warnings in 42.21s
Name                          Stmts   Miss  Cover
pa_core/cli.py                 1107    473    57%
pa_core/llm/compare_runs.py      237     83    65%
```

The final working-tree acceptance command was rerun after the commit blocker
was identified, under a 120-second subprocess timeout:

```text
python -m pytest tests/test_coverage_round_2375.py -q -m "not slow"
12 passed, 139 warnings in 4.00s
exit=0; total subprocess wall time=6.669s
```

This is verified working-tree content, bound to the hashes above, rather than
a claimed final commit or hosted result.

These percentages do not meet a new whole-product coverage target and cannot
establish a 120-unit Gate reduction. This change strengthens observable
correctness on existing CLI paths. It does not claim a coverage reduction for
either ranked target or a completed exception for all their missing lines.
Current-source inspection confirms the two narrow historical exceptions:
`compare_runs._format_value` has an internal `_MISSING` branch while
`_config_snapshot_diff` formats that sentinel directly; CLI sweep entry requires
`not args.sensitivity` while its nested lines 1500–1522 require `args.sensitivity`.
Those narrow exceptions do not explain away all uncovered units.

Hosted acceptance still requires a current-main/final-head pair with matching
collection, Python, dependencies, product sources, branch setting, workers and
coverage flags; download both reports and record per-target reductions or
concrete justified exceptions. The old branch's numbers are historical context
only. `gh pr view codex/issue-2496 --repo
stranske/Portable-Alpha-Extension-Model --json
number,url,isDraft,state,headRefOid` also failed with the same API connection
error. No PR was created or changed in this run, and no hosted result or PR
readiness state is asserted.
Attempts to post the blocker with `gh issue comment 2496 --body-file ...` and
add `needs-human` with `gh issue edit 2496 --add-label needs-human` also failed
with the same API connection error. The blocker is preserved here locally.

The required commit was attempted after tests and formatting passed. Staging
failed before modifying the index:

```text
git add tests/test_coverage_round_2375.py docs/evidence/issue-2496/README.md docs/evidence/issue-2496/mutations.json
fatal: Unable to create '.../.git/index.lock': Read-only file system
```

This environment grants read access to `.git` and cannot approve a permission
escalation. The source and evidence remain in the working tree; an apply-ready
patch is saved at `/tmp/issue-2496.patch`. No commit or push is claimed. A run
with a writable Git directory must commit these changes on the existing branch
and complete the hosted acceptance steps.

Task status (duplicates in the issue refer to the same evidence):

- [ ] Live `gh pr view`/`gh pr diff` reconciliation against fresh main.
- [x] Strengthen completed workbook/manifest bundle integrity; preserve #2380 cases.
- [x] Assert zero GIF animation/export calls after conversion failure through the CLI.
- [x] Record attribution fallback symbol/revision and conditional-fix decision.
- [x] Record pre-edit outcome/command/stop rule, collection and fresh production mutation/restoration evidence for every strengthened case.
- [ ] Final-revision hosted pytest/CI links and configured-check results.
- [ ] Matching hosted Gate baseline/final comparison and per-target coverage acceptance.
