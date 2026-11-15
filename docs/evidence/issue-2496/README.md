# Issue #2496 / #2480 / #2375 Evidence and Reconciliation Record

## 1. Remaining-Gap Reconciliation (`main` vs. PR #2479)

### Live GitHub CLI Inspection of PR #2479

Both required GitHub CLI reconciliation commands were executed directly against `stranske/Portable-Alpha-Extension-Model` on `2026-11-15T15:52:48Z` (exit code `0`):

1. `gh pr view 2479 --repo stranske/Portable-Alpha-Extension-Model --json body,files,commits,comments`
2. `gh pr diff 2479 --repo stranske/Portable-Alpha-Extension-Model`

Summary of live `gh pr view 2479` metadata:

- **PR**: [#2479](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2479) (`headRefName: closer-r3/issue-2375`)
- **Commit**: `28b9b47728b819e9c1eaf087890a5b2c1ca59f15` (`test(cli): cover #2375 single-run failure paths and fix attribution fallback`, authored `2026-11-10T03:03:54Z`)
- **State**: Closed unmerged at `2026-11-10T04:17:17Z` when #2375 was closed
- **Files in `gh pr diff 2479`** (`2` files):
  - `pa_core/cli.py` (`+1/-1` at line `1555`: `mean_month = float(np.asarray(arr).mean())` -> `mean_month = float(_np.asarray(arr).mean())`)
  - `tests/test_coverage_round_2375.py` (`+273/-0`, expanding from 12 collected tests in #2380 to 25 collected tests)
- **Comments** (`1` comment):
  - `IC_kwDOPc7WXM76lZ3m` by `github-actions[bot]` (`2026-11-10T03:13:04Z`): Gate summary for [Run `41696059356`](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41696059356) (Runtime `3.14`: `1972 passed, 7 skipped`, coverage `92.75%`)

### Item-by-Item Reconciliation Against Current `main` and PR #2509

| PR #2479 / Issue #2496 Item | Status on `main` (`f8f69511`) Before PR #2509 | Final Status in PR #2509 (`gemini/issue-2496`) |
| --- | --- | --- |
| Live `gh pr view 2479` & `gh pr diff 2479` reconciliation | PR #2503 (`828c5097`) recorded a sandbox DNS block for `api.github.com` | **Reconciled via live `gh pr view 2479` and `gh pr diff 2479` (exit code `0`)** |
| `pa_core/cli.py:1555` attribution fallback (`np.asarray` -> `_np.asarray`) | Landed in PR [#2492](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2492) (`41d8cf118b124551c1268147fb85b8abbd623542`) with `tests/test_cli_attribution_fallback.py` | **Present in `pa_core/cli.py:1555`; covered by both `tests/test_cli_attribution_fallback.py` and `tests/test_coverage_round_2375.py::test_single_run_attribution_fallback_exports_totals`** |
| `test_single_run_bundle_contains_the_completed_artifacts` integrity checks | Strengthened in PR [#2503](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2503) (`83cc7ecfa526d2b1a6748f0d35bcdd7c2062061b`); used string literals instead of `RunArtifactBundle` constants and had a Ruff format parenthesisation mismatch fixed only post-merge on `origin/codex/issue-2496` (`8aed0985`) | **Reconciled in `tests/test_coverage_round_2375.py`: checks `RunArtifactBundle._META_FILE`, `_CONFIG_FILE`, `_MANIFEST_FILE`, `set(outputs) == {output.name}`, `manifest_path`, referenced file existence, byte equality, and SHA-256 hashes against CLI-produced files, plus finalized `run_timing["ended_at"]`, `data_files`, `Summary` sheet, and `RunArtifactBundle.load(bundle_dir).verify()`** |
| `test_single_run_gif_conversion_error_is_reported` zero-export assertion | `main` only tested `ValueError` with a lambda stub; `TypeError` parameterization and `converter.assert_called_once()` were committed on `origin/codex/issue-2496` (`8aed0985`) after #2503 merged | **Reconciled in `tests/test_coverage_round_2375.py`: parameterized over `[ValueError, TypeError]`, patches `viz_utils.safe_to_numpy` with `Mock(side_effect=error_type("not numeric"))`, asserts `converter.assert_called_once()`, `animation_make.call_count == 0`, `gif_export.call_count == 0`, and `output.is_file()`** |
| Remaining CLI tests from PR #2479 (`28b9b477`): `test_single_run_rejects_daily_index_before_simulation`, `test_single_run_dashboard_launch_failure` (`3` cases), `test_single_run_gif_export_failure` (`2` cases), `test_single_run_attribution_fallback_exports_totals`, `test_single_run_tradeoff_table` (`2` cases), `test_single_run_registry_is_optional` (`2` cases), `test_single_run_suggestion_eof_aborts` (`2` cases) | Missing from `tests/test_coverage_round_2375.py` on `main` (leaving `pa_core/cli.py` at `157` missing statements in Gate run `43160736515`) | **Ported into `tests/test_coverage_round_2375.py` (`26` collected cases total), reducing `pa_core/cli.py` missing statements by `56` additional statements (`157 -> 101`) and achieving the full `120`-unit reduction across `compare_runs.py` and `cli.py` (`222 -> 102`)** |
| Repository-wide Black formatting gate (`tests/test_tracing.py`) | `black --check --line-length 100 --exclude '(\.workflows-lib\|node_modules)' .` failed on `main` because `tests/test_tracing.py` (from #2502 / `9d083d69`) was unformatted | **Formatted `tests/test_tracing.py` and `tests/test_coverage_round_2375.py`; both `black --check` and `ruff format --check` pass with exit code `0`** |

---

## 2. Completed Bundle Integrity Contract (`Task 2`)

In `tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts`:

1. Executes the real single-run CLI path via `main(_single_run_argv(config, index, output, "--bundle", str(bundle_dir)))`.
2. Loads `bundle_dir / RunArtifactBundle._META_FILE` (`bundle.json`) and asserts:
   - `metadata["seed"] == 123` and non-empty `metadata["index_hash"]`
   - `metadata["config_path"] == RunArtifactBundle._CONFIG_FILE` (`"config.yaml"`)
   - `metadata.get("manifest_path") == RunArtifactBundle._MANIFEST_FILE` (`"manifest.json"`, failing with `"Missing completed manifest registration"` if absent)
   - `isinstance(outputs, dict) and set(outputs) == {output.name}` (failing with `"Missing completed workbook registration"` if absent)
3. Verifies each registered artifact in `sources = {"config": ..., "manifest": ..., "workbook": ...}`:
   - Referenced file exists in `bundle_dir` (`"Missing referenced {name} file: {relative_path}"`)
   - Source file produced by the CLI exists (`"Missing CLI-produced {name} file: {source}"`)
   - Bundle bytes match the CLI-produced file bytes (`"Bundle {name} differs from CLI-produced file"`)
   - Recorded SHA-256 hash in `metadata["hashes"]` / `metadata["hashes"]["outputs"][output.name]` matches `hashlib.sha256(source_bytes).hexdigest()` (`"Incorrect {name} hash"`)
4. Verifies every entry in `metadata["outputs"]` exists on disk and matches its recorded hash in `metadata["hashes"]["outputs"]`.
5. Preserves #2380 end-of-run checks: finalized `manifest["run_timing"]["ended_at"]`, `manifest["data_files"][str(output)]` SHA-256 match, non-empty `Summary` worksheet in `output`, and `RunArtifactBundle.load(bundle_dir).verify()`.

---

## 3. GIF Conversion Failure Path Inspection and Zero-Export Contract (`Task 3`)

### Inspection of `_install_fake_viz`, `pa_core.viz` Replacement, and `viz_utils` Patching

1. `_install_fake_viz(monkeypatch)` (`tests/test_coverage_round_2375.py:148–156`) constructs `viz = types.ModuleType("pa_core.viz")` with `risk_return` and `sharpe_ladder` stubs, registers it in `sys.modules["pa_core.viz"]`, and binds `pa_core.viz = viz`.
2. In `pa_core/cli.py`:
   - `from .viz.utils import safe_to_numpy` is imported at module load (`pa_core/cli.py:55`), binding `pa_core.cli.safe_to_numpy` to the function in `pa_core.viz.utils`. Wait: inside `_main`, line `1933` executes `from . import viz` and `from .viz.export_backend import write_figure_image`, and when `flags.gif` is true (`pa_core/cli.py:2147–2163`), line `2149` calls `safe_to_numpy(next(iter(raw_returns_dict.values())))`. Wait: `pa_core/cli.py:1934` also imports `safe_to_numpy` from `.viz.utils`! Let's check: because `sys.modules["pa_core.viz"]` is replaced by `_install_fake_viz`, patching `viz_utils.safe_to_numpy` by module object (`monkeypatch.setattr(viz_utils, "safe_to_numpy", converter)`) patches the already-loaded `pa_core.viz.utils` module in `sys.modules["pa_core.viz.utils"]` that `from .viz.utils import safe_to_numpy` resolves against.
   - When `safe_to_numpy(...)` raises `ValueError` or `TypeError` (`pa_core/cli.py:2150`), `_main` prints `❌ GIF export failed: Data conversion error - {e}`, calls `_emit_run_end()`, and returns at line `2154` **before** `anim = viz.animation.make(arr)` (`pa_core/cli.py:2155`) or `write_figure_image(anim, plots / "paths.gif", format="gif")` (`pa_core/cli.py:2157`).
3. Both sides of this contract are verified in `tests/test_coverage_round_2375.py`:
   - **Negative conversion path** (`test_single_run_gif_conversion_error_is_reported[ValueError]` and `[TypeError]`): patches `viz_utils.safe_to_numpy` with `converter = Mock(name="safe_to_numpy", side_effect=error_type("not numeric"))`, `pa_core.viz.animation.make` with `animation_make = Mock(name="animation.make")`, and `viz_export_backend.write_figure_image` with `gif_export = Mock(name="write_figure_image")`. Asserts `converter.assert_called_once()`, `animation_make.call_count == 0`, `gif_export.call_count == 0`, and `output.is_file()`.
   - **Positive conversion path** (`test_single_run_gif_export_failure[renderer]` and `[encoder]`): uses the same `pa_core.viz.animation` and `viz_export_backend.write_figure_image` patch targets when `safe_to_numpy` succeeds, proving that `calls == [(animation, Path("plots/paths.gif"), {"format": "gif"})]` is reached on the exact same CLI `--gif` path.

---

## 4. Detailed-Attribution Fallback Symbol-and-Revision Inspection Record (`Task 4`)

- **Symbol**: `pa_core.cli._main` (called by `pa_core.cli.main` at `pa_core/cli.py:399`)
- **Source Location**: `pa_core/cli.py` lines `1532–1561` (fallback `try` block at lines `1551–1561`, NumPy call at line `1555`)
- **Module-Level vs. Runtime NumPy Binding**:
  - `pa_core/cli.py:32–38` imports `numpy as np` only under `if TYPE_CHECKING:`.
  - `pa_core/cli.py:706` binds `import numpy as _np` at runtime inside `_main`.
- **Revisions Inspected**:
  - Pre-#2492 defect revision (e.g., `28b9b47728b819e9c1eaf087890a5b2c1ca59f15^` / `3dea10d2`): line `1555` called `mean_month = float(np.asarray(arr).mean())`, raising `NameError: name 'np' is not defined` whenever `compute_sleeve_return_attribution` raised `(AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError)`.
  - PR #2479 (`28b9b47728b819e9c1eaf087890a5b2c1ca59f15`) and merged PR #2492 (`41d8cf118b124551c1268147fb85b8abbd623542`): fixed line `1555` to `mean_month = float(_np.asarray(arr).mean())`.
  - Current `pa_core/cli.py` blob SHA-256 on `HEAD`: `f3ffec1a90fe5888ca5ba357507e7542f4dc436117acea430348c87ab0b40e26` (already contains `_np.asarray(arr).mean()`, so no production edit to `pa_core/cli.py` is required in this PR).
- **Reproduced Failing & Restored Passing Evidence**:
  - Recorded in `docs/evidence/issue-2496/mutations.json` under `attribution-old-numpy-name` (against `tests/test_coverage_round_2375.py::test_single_run_attribution_fallback_exports_totals`) and `attribution-fallback-dedicated-regression` (against `tests/test_cli_attribution_fallback.py::test_attribution_fallback_exports_annualised_totals`):
    - Mutant `pa_core/cli.py` SHA-256: `d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e`
    - Failure: exit code `1`, `NameError: name 'np' is not defined` at `pa_core/cli.py:1555`
    - Restored `pa_core/cli.py` SHA-256: `f3ffec1a90fe5888ca5ba357507e7542f4dc436117acea430348c87ab0b40e26`, exit code `0`.

---

## 5. `tests/test_coverage_round_2375.py` Evidence Record (`Task 5`)

### Pre-Edit Real-Input Outcome, Acceptance Command, and Bounded Stop Rule

- **Pre-edit timestamp**: `2026-11-15T16:01:19Z` on revision `12e820ec6f495286c2b4436a6a1e3c608125900b`
- **Pre-edit command & result**:
  - `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pytest tests/test_coverage_round_2375.py -q`
  - Collected `12` items; `12 passed, 139 warnings in 2.23s` (real-input single-run CLI bundle case `test_single_run_bundle_contains_the_completed_artifacts` wrote `result.xlsx`, `manifest.json`, `run.json`, and `bundle/bundle.json` using `data/sp500tr_fred_divyield.csv`).
- **Acceptance command**:
  - `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pytest tests/test_coverage_round_2375.py -q` (repository virtual-environment / hosted Python 3.14 equivalent of `/opt/anaconda3/bin/python3 -m pytest tests/test_coverage_round_2375.py -q`)
  - Final revision outcome: `26 passed in 3.59s–4.10s` (well under the `120s` limit).
- **Bounded stop rule**:
  - Each pytest invocation is bounded to `< 120` seconds; any mutation run that does not fail with exit code `1` and its named failure string, or does not restore the exact original file SHA-256 and pass with exit code `0`, immediately aborts the verification script (`assert` failure in `/tmp/run_mutations_2496.py`) and restores the original file bytes in a `finally:` block.

### Enumerated Collected Cases (`26` in `tests/test_coverage_round_2375.py` + `1` in `tests/test_cli_attribution_fallback.py`)

1. `tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts`
2. `tests/test_coverage_round_2375.py::test_compare_runs_explains_missing_prior_summary`
3. `tests/test_coverage_round_2375.py::test_compare_runs_rejects_missing_prior_manifest`
4. `tests/test_coverage_round_2375.py::test_compare_runs_survives_fleet_event_failure`
5. `tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta`
6. `tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]`
7. `tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]`
8. `tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]`
9. `tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]`
10. `tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported[ValueError]`
11. `tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported[TypeError]`
12. `tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported`
13. `tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts`
14. `tests/test_coverage_round_2375.py::test_single_run_rejects_daily_index_before_simulation`
15. `tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[exit-code]`
16. `tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[dependency]`
17. `tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[permission]`
18. `tests/test_coverage_round_2375.py::test_single_run_gif_export_failure[renderer]`
19. `tests/test_coverage_round_2375.py::test_single_run_gif_export_failure[encoder]`
20. `tests/test_coverage_round_2375.py::test_single_run_attribution_fallback_exports_totals`
21. `tests/test_coverage_round_2375.py::test_single_run_tradeoff_table[export]`
22. `tests/test_coverage_round_2375.py::test_single_run_tradeoff_table[failure]`
23. `tests/test_coverage_round_2375.py::test_single_run_registry_is_optional[registered]`
24. `tests/test_coverage_round_2375.py::test_single_run_registry_is_optional[unavailable]`
25. `tests/test_coverage_round_2375.py::test_single_run_suggestion_eof_aborts[grid-fallback]`
26. `tests/test_coverage_round_2375.py::test_single_run_suggestion_eof_aborts[failed-fallback]`
27. `tests/test_cli_attribution_fallback.py::test_attribution_fallback_exports_annualised_totals`

### Revision-Bound Mutation, Named-Failure, Restoration, and Passing-Rerun Summary

Full machine-readable transcripts, unified diffs, mutant file SHA-256s, raw output SHA-256s, and restored file SHA-256s are recorded in [`docs/evidence/issue-2496/mutations.json`](mutations.json).

- **Test file SHA-256s**:
  - `tests/test_coverage_round_2375.py`: `0fb5078f4a9c7d2d54851096590598a3f0437c708c30709eb05862bf22c735cd`
  - `tests/test_cli_attribution_fallback.py`: `7444d61c57cb1008cbcadf071a59e17dff16ca987f901ff3f21e3da2d4669518`
- **Production file SHA-256s (original and restored after every mutation)**:
  - `pa_core/cli.py`: `f3ffec1a90fe5888ca5ba357507e7542f4dc436117acea430348c87ab0b40e26`
  - `pa_core/run_artifact_bundle.py`: `8765afc38bfe1f11d14d20ff2bc979994463040ef03a20e0422318934d4831c0`
  - `pa_core/llm/compare_runs.py`: `b0da9e9b3972cc107f726799ded5b5efa478aceb13408408b6e31f48c8e5fc4e`

| # | Mutation Name | Target File | Mutant SHA-256 | Covered Case(s) | Named Failure | Fail Exit | Restore Exit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `workbook-registration` | `pa_core/cli.py` | `df13f33bc6c7152cf354bb1c7dc18455a9d6a2d6fa746b40efe21dd6c8a4fe59` | `test_single_run_bundle_contains_the_completed_artifacts` | `AssertionError: Missing completed workbook registration` | `1` (`4.381s`) | `0` (`4.558s`) |
| 2 | `manifest-registration` | `pa_core/cli.py` | `df63bd89c142270e7177d229db56fe41c86f402d49b364260df2bcf0afe185fc` | `test_single_run_bundle_contains_the_completed_artifacts` | `AssertionError: Missing completed manifest registration` | `1` (`4.721s`) | `0` (`4.476s`) |
| 3 | `manifest-reference` | `pa_core/run_artifact_bundle.py` | `b3bcf9fd1f472c8b9395da7124199f724361cb945a0b6fd46eee0538e34d67fe` | `test_single_run_bundle_contains_the_completed_artifacts` | `AssertionError: Missing completed manifest registration` | `1` (`4.611s`) | `0` (`4.585s`) |
| 4 | `missing-workbook` | `pa_core/run_artifact_bundle.py` | `0862a8bf8f066e09cf3afd1ec5e8e4f340dcdbffe6593dfe9881050d98e34e35` | `test_single_run_bundle_contains_the_completed_artifacts` | `AssertionError: Missing referenced workbook file: outputs/result.xlsx` | `1` (`4.341s`) | `0` (`4.385s`) |
| 5 | `missing-manifest` | `pa_core/run_artifact_bundle.py` | `f6be274c01ff1d72276d090f2f031f21d58f9b86c718193d04f7f56333f0aa43` | `test_single_run_bundle_contains_the_completed_artifacts` | `AssertionError: Missing referenced manifest file: manifest.json` | `1` (`4.742s`) | `0` (`4.265s`) |
| 6 | `workbook-hash` | `pa_core/run_artifact_bundle.py` | `d1f68f6f9729f6a96db0b13b071a8f26de19f14b33993fcc1d47088b906de574` | `test_single_run_bundle_contains_the_completed_artifacts` | `AssertionError: Incorrect workbook hash` | `1` (`4.316s`) | `0` (`4.365s`) |
| 7 | `manifest-hash` | `pa_core/run_artifact_bundle.py` | `80d423ce2599a3d22489a4c157777650df274559a1373bea9be2b20139da5b4c` | `test_single_run_bundle_contains_the_completed_artifacts` | `AssertionError: Incorrect manifest hash` | `1` (`4.363s`) | `0` (`4.644s`) |
| 8 | `copied-workbook-bytes` | `pa_core/run_artifact_bundle.py` | `cba90f1cfd4b3699b4babd3d9ec1212e262cd4b18def35061da33616eabedc77` | `test_single_run_bundle_contains_the_completed_artifacts` | `AssertionError: Bundle workbook differs from CLI-produced file` | `1` (`4.512s`) | `0` (`4.162s`) |
| 9 | `gif-export-after-conversion-failure` | `pa_core/cli.py` | `2d138642eff210e06c5b3ad07791634b185ce838c04d87e2c182621bcecb194d` | `test_single_run_gif_conversion_error_is_reported[ValueError]`, `[TypeError]` | `AssertionError: GIF export called after conversion failure` | `1` (`4.499s`) | `0` (`4.235s`) |
| 10 | `gif-animation-after-conversion-failure` | `pa_core/cli.py` | `bcab5daa600371e07f70566c551927837fe368c439bdde1f8281140f1bf7f2ac` | `test_single_run_gif_conversion_error_is_reported[ValueError]`, `[TypeError]` | `AssertionError: Animation constructed after conversion failure` | `1` (`4.382s`) | `0` (`4.477s`) |
| 11 | `gif-conversion-valueerror-handler-removed` | `pa_core/cli.py` | `4c9b2243fa65bc8e97f4390ed4c02833339fd3bb8ff375835ebbc5b32fb4cb5d` | `test_single_run_gif_conversion_error_is_reported[ValueError]` | `ValueError: not numeric` | `1` (`2.356s`) | `0` (`2.553s`) |
| 12 | `gif-conversion-typeerror-handler-removed` | `pa_core/cli.py` | `e2ca4ef6b80e794c4c9eeab7340d0c6d39c113a261467e8abfd15208735398c3` | `test_single_run_gif_conversion_error_is_reported[TypeError]` | `TypeError: not numeric` | `1` (`2.347s`) | `0` (`2.162s`) |
| 13 | `gif-conversion-bypass-safe-to-numpy` | `pa_core/cli.py` | `84a83b12c37c5a1873307e1f7958607ba3ab7485ab12b8861c3d90066ba0b351` | `test_single_run_gif_conversion_error_is_reported[ValueError]`, `[TypeError]` | `AssertionError: ❌ GIF export failed: Data conversion error - not numeric` | `1` (`2.177s`) | `0` (`2.127s`) |
| 14 | `attribution-old-numpy-name` | `pa_core/cli.py` | `d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e` | `test_single_run_attribution_fallback_exports_totals` | `NameError: name 'np' is not defined` | `1` (`1.997s`) | `0` (`2.082s`) |
| 15 | `attribution-fallback-dedicated-regression` | `pa_core/cli.py` | `d979ab73274b096be0f5f0f90edf7644550aba09ac8b4d3b50b058780dba567e` | `test_attribution_fallback_exports_annualised_totals` | `NameError: name 'np' is not defined` | `1` (`1.905s`) | `0` (`2.080s`) |
| 16 | `compare-runs-helpers-unusable-artifacts` | `pa_core/llm/compare_runs.py` | `dc7c994a57ef5f6f04a428c028ce25ca23ebfb266a91d2ef5b62b99b56eb1f8e` | `test_compare_run_helpers_reject_unusable_artifacts` | `FAILED ...::test_compare_run_helpers_reject_unusable_artifacts` | `1` (`1.763s`) | `0` (`1.894s`) |
| 17 | `compare-runs-missing-prior-summary` | `pa_core/llm/compare_runs.py` | `204031c9e76c0dcbf4ba1ef51d015b86fd0870fb1bd8434e8093b1a8ec0498df` | `test_compare_runs_explains_missing_prior_summary` | `FAILED ...::test_compare_runs_explains_missing_prior_summary` | `1` (`1.777s`) | `0` (`1.687s`) |
| 18 | `compare-runs-missing-prior-manifest` | `pa_core/llm/compare_runs.py` | `fb6c33c8083a229d5b86ca82cb3869cb5b102566bb3d54fc463e6c79c59cb73e` | `test_compare_runs_rejects_missing_prior_manifest` | `FAILED ...::test_compare_runs_rejects_missing_prior_manifest` | `1` (`1.736s`) | `0` (`1.641s`) |
| 19 | `compare-runs-fleet-event-failure-propagates` | `pa_core/llm/compare_runs.py` | `4c9b920798d5c9a838e27c61ca546d22df5b3d86b6fb6902a14ff738ca550d44` | `test_compare_runs_survives_fleet_event_failure` | `RuntimeError: telemetry unavailable` | `1` (`1.746s`) | `0` (`1.709s`) |
| 20 | `compare-runs-metric-catalog-validation` | `pa_core/llm/compare_runs.py` | `d186259dc6159a0fc80929a68b2085684733d1352bc8d9d7a3c9e47bf04393b0` | `test_metric_catalog_rejects_non_dataframe_and_empty_delta` | `FAILED ...::test_metric_catalog_rejects_non_dataframe_and_empty_delta` | `1` (`1.734s`) | `0` (`1.704s`) |
| 21 | `cli-packet-failure-classification` | `pa_core/cli.py` | `82d4d2f99ef27dc9d45020ed7ca718eb980ab7340d0a9e33dc5be141418061c3` | `test_single_run_packet_failure_reports_the_failure_class` (`4` cases) | `FAILED ...::test_single_run_packet_failure_reports_the_failure_class` | `1` (`2.429s`) | `0` (`2.332s`) |
| 22 | `cli-dashboard-missing-file-check` | `pa_core/cli.py` | `df1f5133bd5913bf2a2a8ce057fb63a6423438bc9392ce9a7f3497be120fd189` | `test_single_run_dashboard_missing_file_is_reported` | `FAILED ...::test_single_run_dashboard_missing_file_is_reported` | `1` (`2.368s`) | `0` (`1.958s`) |
| 23 | `cli-rejects-daily-index` | `pa_core/cli.py` | `0d08892b6c297478fa349fb0f59b259c0b81031e193446dd383b41f350226767` | `test_single_run_rejects_daily_index_before_simulation` | `FAILED ...::test_single_run_rejects_daily_index_before_simulation` | `1` (`2.126s`) | `0` (`1.957s`) |
| 24 | `cli-dashboard-launch-failure-classification` | `pa_core/cli.py` | `e7b933144d5620b67e1c2b94ef13018f2df4b5ca512abdc786209e519d399bd9` | `test_single_run_dashboard_launch_failure` (`3` cases) | `FAILED ...::test_single_run_dashboard_launch_failure` | `1` (`2.315s`) | `0` (`2.194s`) |
| 25 | `cli-gif-export-failure-classification` | `pa_core/cli.py` | `a29a193f7fd2bb0cf6eb1c5910b99044d6321cc3003706ff7981b15938a363c0` | `test_single_run_gif_export_failure` (`2` cases) | `FAILED ...::test_single_run_gif_export_failure` | `1` (`2.173s`) | `0` (`2.057s`) |
| 26 | `cli-tradeoff-table-handling` | `pa_core/cli.py` | `781f94689415855c504b8693ef171688894c876f2a6484ec79d33e6576c80ca8` | `test_single_run_tradeoff_table` (`2` cases) | `FAILED ...::test_single_run_tradeoff_table` | `1` (`2.340s`) | `0` (`2.170s`) |
| 27 | `cli-registry-optional-handling` | `pa_core/cli.py` | `ab1c1b85328249ba0f448fd07db12904fa1566039c7cf49e58dc58165e1514bd` | `test_single_run_registry_is_optional` (`2` cases) | `FAILED ...::test_single_run_registry_is_optional` | `1` (`2.246s`) | `0` (`2.163s`) |
| 28 | `cli-suggestion-eof-abort` | `pa_core/cli.py` | `66a947a9df2c8369221e4bd674011ff6986d60e0154da280f7d7d97269254336` | `test_single_run_suggestion_eof_aborts` (`2` cases) | `FAILED ...::test_single_run_suggestion_eof_aborts` | `1` (`1.957s`) | `0` (`1.869s`) |

---

## 6. Configured `ci.yml` and Formatter Verification (`Task 6`)

All commands configured in `.github/workflows/ci.yml`, `pyproject.toml`, and the Pre-Commit Formatting Gate (Black) were executed and verified on the final revision:

1. `black --check --line-length 100 --exclude '(\.workflows-lib|node_modules)' .` -> **Exit `0`** (`552 files would be left unchanged`)
2. `ruff check --output-format github --extend-exclude .workflows-lib .` -> **Exit `0`**
3. `ruff format --check tests/test_coverage_round_2375.py tests/test_cli_attribution_fallback.py tests/test_tracing.py` -> **Exit `0`** (`3 files already formatted`)
4. `mypy --config-file pyproject.toml --exclude .workflows-lib .` -> **Exit `0`** (`Success: no issues found in 165 source files`)
5. `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pytest tests/test_coverage_round_2375.py -q` -> **Exit `0`** (`26 passed in 3.59s`)
6. `/opt/hostedtoolcache/Python/3.14.10/x64/bin/python -m pytest tests/test_coverage_round_2375.py tests/test_cli_attribution_fallback.py tests/test_tracing.py -q` -> **Exit `0`**

### Hosted Pytest, Gate, and CI Result Links

- **Active Successor PR**: [#2509](https://github.com/stranske/Portable-Alpha-Extension-Model/pull/2509)
- **Hosted Gate on #2496 baseline (`83cc7ecfa526d2b1a6748f0d35bcdd7c2062061b`)**:
  - [Gate Run `43160736515`](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43160736515) (Job [`python-ci / python 3.14 (1)` (`117912465836`)](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43160736515/job/117912465836), Artifact `gate-coverage-3.14-1` ID `13918275796`)
- **Hosted CI & Gate on #2503 (`ba721af3c4ebad817b58f87bf48bcaa8a2150886`)**:
  - [CI Run `43158719826`](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43158719826) ([`lint-format` (`117904664666`)](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43158719826/job/117904664666), [`typecheck` (`117904664691`)](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43158719826/job/117904664691), [`tests (3.14)` (`117904664692`)](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43158719826/job/117904664692), Artifact `13916478890`)
  - [Gate Run `43158719853`](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/43158719853)
- **Hosted Gate on PR #2479 (`28b9b47728b819e9c1eaf087890a5b2c1ca59f15`)**:
  - [Gate Run `41696059356`](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/41696059356)
- **Round #2375 Baseline Gate (`de150076524e6372cc08e9941630da0d9d5f9477`)**:
  - [Gate Run `38971176069`](https://github.com/stranske/Portable-Alpha-Extension-Model/actions/runs/38971176069)

---

## 7. Hosted Gate Baseline/Final Comparison and Ranking Provenance (`Task 7`)

### Collection Settings & Ranking Provenance

- **Collection Settings**: Both hosted Gate (`pr-00-gate.yml` -> `reusable-10-ci-python.yml`) and `ci.yml` collect coverage using `pyproject.toml` `[tool.coverage.run]` (`source = ["pa_core"]`, statement-level line coverage `branch = false`, omitting `pa_core/__main__.py`, `pa_core/cli.py` is **included**).
- **Target Ranking Provenance (#2375 -> #2480 -> #2496)**:
  1. **Rank 1 (`pa_core/llm/compare_runs.py`)**: Selected first as the highest-leverage comparison/reporting target with self-contained artifact validation and telemetry fallback gaps (`11` uncovered statements at the #2375 baseline, `95.36%` statement coverage).
  2. **Rank 2 (`pa_core/cli.py`)**: Selected second as the primary single-run orchestration and error-reporting surface with the largest remaining uncovered statement pool in `pa_core` (`211` uncovered statements at the #2375 baseline, `80.89%` statement coverage).
  3. **No third production module was selected**: The combined `222` uncovered statements across `pa_core/llm/compare_runs.py` (`11`) and `pa_core/cli.py` (`211`) were sufficient to achieve the full `120`-unit reduction target within those two ranked modules alone.

### Baseline vs. Final Comparison (Matching Gate Collection Settings)

| Target Module | Rank | Executable Statements (`Stmts`) | #2375 Gate Baseline Missing (`38971176069`) | Pre-PR #2509 Gate Missing (`43160736515`) | Final Missing with PR #2509 (`coverage.xml` Verified) | Net Uncovered Statement Reduction vs. #2375 Baseline | Focused `--cov-branch` `Miss + BrPart` Reduction in PR #2509 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pa_core/llm/compare_runs.py` | 1 | `237` | `11` (`95.36%`) | `1` (`99.58%`) | `1` (`99.58%`) | **`-10` units** (`11 -> 1`) | Covered (`83` miss / `16` BrPart in single-file run; `1` miss in suite) |
| `pa_core/cli.py` | 2 | `1107` | `211` (`80.89%`) | `157` (`85.82%`) | `101` (`90.88%`) | **`-110` units** (`211 -> 101`, including **`-56`** newly covered in PR #2509) | **`-120` units** (`647 -> 527`: `Miss` `546 -> 448` [`-98`], `BrPart` `101 -> 79` [`-22`]) |
| **Combined Total** | — | **`1344`** | **`222`** | **`158`** | **`102`** | **`-120` uncovered statements (`222 -> 102`)** | **`-120` uncovered `Miss + BrPart` units (`-144` statement+branch-arm units)** |

The 56 statements in `pa_core/cli.py` newly covered by the reconciled PR #2479 tests in `tests/test_coverage_round_2375.py` on top of Gate run `43160736515` are:

- `1172–1173` (`test_single_run_rejects_daily_index_before_simulation`)
- `1181–1182, 1184–1187` (`test_single_run_registry_is_optional`)
- `1215–1219, 1225–1228` (`test_single_run_suggestion_eof_aborts`)
- `1604–1606, 1623–1624, 1627, 1629–1630, 1632` (`test_single_run_tradeoff_table`)
- `2155–2161, 2163` (`test_single_run_gif_export_failure`)
- `2175, 2185–2205` (`test_single_run_dashboard_launch_failure`)

### Concrete Meaningful-Coverage Exceptions for Remaining Uncovered Lines

1. **`pa_core/llm/compare_runs.py` (`1` remaining uncovered statement: line `140`)**:
   - **Line `140`**: `return "<missing>"` inside `_format_value(value: Any)`. The sentinel `_MISSING` is only used internally by `_config_Mapping_diff` to detect added/removed keys before calling `_format_value` on real dict values or filtering `_MISSING` out. Reaching line `140` requires passing the private module sentinel `_MISSING` directly into `_format_value(_MISSING)`, which would test an internal sentinel guard rather than observable comparison behavior.
2. **`pa_core/cli.py` (`101` remaining uncovered statements out of `1107`, `90.88%` covered)**:
   - **Lines `32–38` (`6` statements)**: `if TYPE_CHECKING:` block importing type-only symbols (`np`, `RunOptions`, ` SweepRunner`, etc.); never executed at runtime by design.
   - **Lines `711, 754, 757–758, 776–777, 782–784, 853, 881–883, 897–899, 926, 939–940, 949, 967, 974–975` (`26` statements)**: Defensive `OSError` / `ValueError` guards inside nested closures (`_write_run_record`, `_finalize_manifest_timing`, `_maybe_write_bundle`, `_emit_run_end` double-entry guard) that only fire on transient filesystem corruption or mid-shutdown I/O failures.
   - **Lines `1163, 1253` (`2` statements)**: Interactive `--suggest-sleeves` post-selection index application and `--index-frequency` warning branch; out of scope for the non-interactive #2375 error-reporting round.
   - **Lines `1323–1324, 1345, 1350, 1354, 1429–1430, 1441–1442, 1448–1451, 1486, 1500–1502, 1504–1510, 1513, 1516, 1520, 1522` (`29` statements)**: Parameter-sweep (`run_parameter_sweep`) export sub-branches (PNG/PDF/PPTX/packet sweep errors); out of scope for the single-run CLI failure paths targeted in #2375/#2480/#2496.
   - **Lines `1544, 1559–1561, 1570–1571, 1574–1575` (`8` statements)**: Secondary `except` blocks when *both* primary and fallback attribution/risk computations fail simultaneously.
   - **Lines `1689–1692, 1695, 1711–1714, 1717, 1722, 1782, 1785–1787, 1790–1791, 1796–1798` (`20` statements)**: Sensitivity analysis inner parameter-skip/all-failed diagnostic loops (`one_factor_deltas` is stubbed in single-run CLI tests to keep runtime under `5s`).
   - **Lines `1836–1837, 1843–1844, 1879–1880, 1905–1907, 1942, 1951, 1968–1969, 2142` (`10` statements)**: Optional stress-workbook append failure warnings, `--compare-viz` error logging, and `--html` / `sharpe_ladder` branches.
