# Issue #2516 manifest output path normalization deliberate-break evidence

Reproduces the acceptance gate on branch `gemini/issue-2516`.

- Issue: #2516
- Named gate: `tests/test_llm_compare_runs.py::test_cli_nested_relative_output_round_trips_to_loader`
- Captured: `2026-11-17T02:49:01Z`

## Baseline (with `artifact_paths` normalization)

```text
python3 -m pytest tests/test_llm_compare_runs.py::test_cli_nested_relative_output_round_trips_to_loader -q
.                                                                        [100%]
1 passed, 16 warnings in 1.18s

python3 -m pytest tests/test_llm_compare_runs.py -q
.......................................                                  [100%]
39 passed, 34 warnings in 1.62s
```

## Deliberate break

Omitted `artifact_paths` manifest-relative normalization in `pa_core/manifest.py` and `pa_core/cli.py`:

```diff
--- a/pa_core/manifest.py
+++ b/pa_core/manifest.py
@@ -121,9 +121,5 @@ class ManifestWriter:
         normalized_artifacts = (
-            {
-                str(key): self._manifest_relative_path(value)
-                for key, value in artifact_paths.items()
-                if value is not None
-            }
+            {str(key): str(value) for key, value in artifact_paths.items() if value is not None}
             if artifact_paths is not None
             else None
         )
--- a/pa_core/cli.py
+++ b/pa_core/cli.py
@@ -1317,2 +1317,2 @@ def main(argv: Sequence[str] | None = None, *, deps: Dependencies | None = None)
             data_quality=idx_series.attrs.get("data_quality"),
-            artifact_paths={"output": str(output_path.relative_to(manifest_json.parent))},
+            artifact_paths={"output": str(output_path)},
         )
@@ -1872,2 +1872,2 @@ def main(argv: Sequence[str] | None = None, *, deps: Dependencies | None = None)
             data_quality=idx_series.attrs.get("data_quality"),
-            artifact_paths={"output": str(out_path.relative_to(manifest_json.parent))},
+            artifact_paths={"output": str(out_path)},
         )
```

With normalization omitted, `load_prior_summary` joined `runs/prior/Outputs.xlsx` onto `manifest_path.parent` (`runs/prior`) to produce `runs/prior/runs/prior/Outputs.xlsx`, and the named gate failed:

```text
python3 -m pytest tests/test_llm_compare_runs.py::test_cli_nested_relative_output_round_trips_to_loader -q
F                                                                        [100%]
=================================== FAILURES ===================================
____________ test_cli_nested_relative_output_round_trips_to_loader _____________
>       assert resolved_path == manifest_path.parent / "Outputs.xlsx"
E       AssertionError: assert PosixPath('/tmp/pytest-of-runner/pytest-4/test_cli_nested_relative_outpu0/runs/prior/runs/prior/Outputs.xlsx') == (PosixPath('/tmp/pytest-of-runner/pytest-4/test_cli_nested_relative_outpu0/runs/prior') / 'Outputs.xlsx')
E        +  where PosixPath('/tmp/pytest-of-runner/pytest-4/test_cli_nested_relative_outpu0/runs/prior') = PosixPath('/tmp/pytest-of-runner/pytest-4/test_cli_nested_relative_outpu0/runs/prior/manifest.json').parent
FAILED tests/test_llm_compare_runs.py::test_cli_nested_relative_output_round_trips_to_loader
1 failed, 16 warnings in 1.28s
```

## Restoration

Reverted the temporary break so `ManifestWriter.write` and `pa_core/cli.py` persist manifest-relative `artifact_paths['output']` (`"Outputs.xlsx"`) while preserving `cli_args['output']` (`"runs/prior/Outputs.xlsx"`). Both the named test and the full `tests/test_llm_compare_runs.py` file passed again (see Baseline).
