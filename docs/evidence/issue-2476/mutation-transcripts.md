# Production-mutation transcripts for issue #2476

Regenerated 2026-11-10 at code revision `54219bd739793fa754c3fec319bb2c6b8f8dd898` with
`.venv/bin/python` (Python 3.14.3). For each record a harness read the production file, recorded
its sha256, wrote the mutation, ran the named command (red), wrote the original bytes back,
checked the sha256 again, and reran the command (green). No test expectation was changed to force
a failure.

## Audit of the earlier records

The 39 records committed in 1479e509 were re-applied to production bytes:

- 21 applied as recorded and still go red then green.
- 16 did not apply byte-for-byte (trailing blank diff lines, a missing trailing comma, wrong
  indentation for the Series branch, or text that occurs twice in facade.py). After the whitespace
  was normalized they go red then green. The two-occurrence index-frame mutation is now two records,
  one for run_single and one for run_sweep.
- 2 were ineffective and are replaced. The scalar tuple/list record did not fail the original
  test: its listed red signal was `assert [1, 2] == [1, 2]`. The column-mapping record mutated the
  row branch, which that input never reaches. The tests now use numpy scalars and non-finite values,
  the tuple and list branches are mutated separately, and the column-mapping mutation targets the
  `_records_to_builtin(records)` assignment at `pa_core/facade.py:143`.
- Several earlier red signals did not match the real output (for example ndarray, empty-mapping and
  keep-mapping). The signals below are the ones these runs printed.

## Mixed Series/dict facade fix

Record `test_serialize_agent_semantics_from_series_and_dict_list` is the fix's original failure.
Its mutated facade bytes are byte-identical to `origin/main:pa_core/facade.py`
(sha256 `f41c355c331e5c6eaa308df594e6908eb7ec7c5e1321d3e165373a1151a2bc07`). On those bytes the test
fails with `AttributeError: 'dict' object has no attribute 'dtype'`. On the fixed bytes
(sha256 `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`) it passes.

## `test_redact_sensitive_ignores_blank_and_non_string_secrets`

- File: `pa_core/llm/result_explain.py`
- Record: recorded
- sha256 before: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841`
- sha256 mutated: `ddd9e4ffb2a0dcd02e52583b05c8dc8d3cd6b05681e766319db6bf3ab7acd937`
- sha256 restored: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert 'prefix [REMOVED] suffix' == 'prefix [REDACTED] suffix'`
- Red summary: `1 failed, 14 warnings in 3.33s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.31s`

Mutation:

```diff
-_REDACTION_TOKEN = "[REDACTED]"
+_REDACTION_TOKEN = "[REMOVED]"
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_redact_sensitive_ignores_blank_and_non_string_secrets
```

## `test_extract_response_text_uses_response_string_when_content_is_not_text`

- File: `pa_core/llm/result_explain.py`
- Record: recorded
- sha256 before: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841`
- sha256 mutated: `cda8a70c1af3efc06646118e8ec6782122012e4908bdc21b4a71bdd6c655b218`
- sha256 restored: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert '' == 'namespace(content=None)'`
- Red summary: `1 failed, 14 warnings in 3.34s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.36s`

Mutation:

```diff
-    return str(response).strip()
+    return ""
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_extract_response_text_uses_response_string_when_content_is_not_text
```

## `test_extract_response_text_skips_items_that_are_not_text_or_mappings`

- File: `pa_core/llm/result_explain.py`
- Site: `pa_core/llm/result_explain.py:285`, inserted as a new `elif` after
  the `Mapping` branch in the `_extract_response_text` content loop (lines
  279-285); this is an addition-only mutation with no removed line
- Record: recorded (diff whitespace normalized)
- sha256 before: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841`
- sha256 mutated: `a8ae3ff3881ecedbaae64b438e5391ee66d7ed183ec93803e8817091e5c57f86`
- sha256 restored: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert 'keep 5 tail' == 'keep tail'`
- Red summary: `1 failed, 14 warnings in 3.38s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.37s`

Mutation:

```diff
+            elif not isinstance(item, Mapping):
+                chunks.append(str(item))
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_extract_response_text_skips_items_that_are_not_text_or_mappings
```

## `test_extract_metric_delta_ignores_non_mapping_summary_values`

- File: `pa_core/llm/result_explain.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841`
- sha256 mutated: `01c3fd00ee7d77bb49282cef2c916f166017b605238dd666162144dcb252cce9`
- sha256 restored: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841` (matches)
- Red exit: 1
- Red signal: `E   AttributeError: 'str' object has no attribute 'get'`
- Red summary: `1 failed, 14 warnings in 3.39s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.39s`

Mutation:

```diff
-        if isinstance(item, Mapping):
+        if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_extract_metric_delta_ignores_non_mapping_summary_values
```

## `test_pa_sweep_omits_substream_metadata_when_runner_has_none`

- File: `pa_core/pa.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9`
- sha256 mutated: `ee93d966a6c02daa638d6b668b945263bc5c21957ec12af941d1d79a3d5b21af`
- sha256 restored: `5e038fb6a878c8d482f7ca1856ca74a4e7fd1b71099a7517025dba2fd153ddb9` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert {'rng_seed': ...ream_ids': []} == {'rng_seed': None}`
- Red summary: `1 failed, 14 warnings in 3.55s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.71s`

Mutation:

```diff
-        if runner.substream_ids:
+        if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_pa_sweep_omits_substream_metadata_when_runner_has_none
```

## `test_run_single_and_sweep_accept_single_column_index_frame (run_single occurrence)`

- File: `pa_core/facade.py`
- Record: recorded text occurs in run_single and run_sweep; each occurrence mutated separately
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `a0dd401781293f8e7bb4b5435087f8e37023d0836ac29fa067d89cf41dce2e20`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   ValueError: Index data must be convertible to pandas Series`
- Red summary: `1 failed, 14 warnings in 3.38s`
- Green exit: 0
- Green signal: `1 passed, 15 warnings in 3.43s`

Mutation:

```diff
-        if not isinstance(idx_series, pd.Series):
+        if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_run_single_and_sweep_accept_single_column_index_frame
```

## `test_run_single_and_sweep_accept_single_column_index_frame (run_sweep occurrence)`

- File: `pa_core/facade.py`
- Record: recorded text occurs in run_single and run_sweep; each occurrence mutated separately
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `4ec687e2d9ecdbdc583c4948265811699ce2ffee53417bdc0e6629ad2048f525`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   ValueError: Index data must be convertible to pandas Series`
- Red summary: `1 failed, 15 warnings in 3.41s`
- Green exit: 0
- Green signal: `1 passed, 15 warnings in 3.42s`

Mutation:

```diff
-        if not isinstance(idx_series, pd.Series):
+        if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_run_single_and_sweep_accept_single_column_index_frame
```

## `test_run_single_continues_when_sleeve_validation_reports_no_violations`

- File: `pa_core/facade.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `467b6a2a801028f68122265e0da7057021b611a8abe72aa8c5e2fbb19ad59900`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   ValueError: Sleeve constraint violations:`
- Red summary: `1 failed, 14 warnings in 3.52s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.53s`

Mutation:

```diff
-        if violations:
+        if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_run_single_continues_when_sleeve_validation_reports_no_violations
```

## `test_run_single_records_repair_info_when_correlation_matrices_absent`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `d27ae058226e6d983c824369577fdb49e833e6fa0c635297d87da55c4a2217ab`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert 'broken' == 'absent'`
- Red summary: `1 failed, 14 warnings in 3.37s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.37s`

Mutation:

```diff
-                    "mode": corr_repair_info.get("repair_mode"),
+                    "mode": "broken",
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_run_single_records_repair_info_when_correlation_matrices_absent
```

## `test_run_single_skips_correlation_frames_when_matrix_shapes_differ`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `653285e6a69e00b0f3ac80bc33ac1428069c92c5e7229d903ee9a887bb141e52`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert '_corr_before_df' not in {'backend': 'numpy', 'N_SIMULATIONS': 2, 'N_MONTHS': 3, 'return_unit': 'monthly', ...}`
- Red summary: `1 failed, 14 warnings in 3.40s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.37s`

Mutation:

```diff
-            if corr_before.shape == corr_after.shape and corr_before.size:
+            if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_run_single_skips_correlation_frames_when_matrix_shapes_differ
```

## `test_export_sweep_without_manifest_passes_no_metadata`

- File: `pa_core/facade.py`
- Site: `pa_core/facade.py:784`, the `SweepArtifacts` branch (second of two
  `if artifacts.manifest is not None:` occurrences; the first, at line 767, is
  the `RunArtifacts` branch and is not mutated)
- Record: recorded (diff whitespace normalized)
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `41c2264ff64dfa2993d9dc2ffb49cde925e1ad7ede3192a66dc53b7c4b0ba02a`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AttributeError: 'NoneType' object has no attribute 'get'`
- Red summary: `1 failed, 14 warnings in 3.71s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.55s`

Mutation:

```diff
-        if artifacts.manifest is not None:
+        if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_export_sweep_without_manifest_passes_no_metadata
```

## `test_export_normalizes_tuple_agent_semantics_to_a_list`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `1132e85d863186082a6680e736ab23744a52c6ff267f66dece7c1fbd8bd79d8d`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert <class 'numpy.int64'> is int`
- Red summary: `1 failed, 14 warnings in 3.62s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.60s`

Mutation:

```diff
-inputs["_agent_semantics_df"] = _records_to_builtin(list(agent_semantics_val))
+inputs["_agent_semantics_df"] = list(agent_semantics_val)
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_export_normalizes_tuple_agent_semantics_to_a_list
```

## `test_render_explain_results_forwards_summary_manifest_and_workbook`

- File: `dashboard/pages/4_Results.py`
- Record: recorded
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `f0d5e17f9467a4433bb335ffe5f795c24225f0f72abdfe505ccfe5b4dfa389b5`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert 'broken.xlsx' == 'results.xlsx'`
- Red summary: `1 failed, 14 warnings in 3.40s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.36s`

Mutation:

```diff
-render_explain_results_panel(summary_df=summary, manifest=manifest_data, xlsx_path=xlsx)
+render_explain_results_panel(summary_df=summary, manifest=manifest_data, xlsx_path="broken.xlsx")
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_render_explain_results_forwards_summary_manifest_and_workbook
```

## `test_results_main_drops_manifest_when_json_cannot_be_parsed`

- File: `dashboard/pages/4_Results.py`
- Site: `dashboard/pages/4_Results.py:214`, the `except Exception:` guarding
  the top-level manifest load in `main()` (not the occurrences at lines 160
  or 360)
- Record: recorded
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `be3d34b746fc29cda526860ed0aa09b25d8a6cc70cf4375a3def10d54a1aa7bc`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)`
- Red summary: `1 failed, 14 warnings in 3.39s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.54s`

Mutation:

```diff
-        except Exception:
+        except ImportError:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_drops_manifest_when_json_cannot_be_parsed
```

## `test_results_main_omits_seed_caption_when_manifest_seed_is_absent`

- File: `dashboard/pages/4_Results.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `239479e0ce7be1cee49a8d850902a083eb3462d45e0f2edc71b2b641d92d427b`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert ['Seed: None'] == []`
- Red summary: `1 failed, 14 warnings in 3.56s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.54s`

Mutation:

```diff
-            if seed is not None:
+            if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_omits_seed_caption_when_manifest_seed_is_absent
```

## `test_results_main_omits_tracking_error_when_neither_column_is_present`

- File: `dashboard/pages/4_Results.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `f0587649ed62e4b06308ee88b12094d22d3f113df48002f3ec3c7556f37da5bb`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   assert False`
- Red summary: `1 failed, 14 warnings in 3.55s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.55s`

Mutation:

```diff
-    if SUMMARY_TE_COLUMN in summary:
-        te_column = SUMMARY_TE_COLUMN
+    if True:
+        te_column = SUMMARY_CVAR_COLUMN
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_omits_tracking_error_when_neither_column_is_present
```

## `test_results_main_skips_non_headline_charts_when_paths_are_missing`

- File: `dashboard/pages/4_Results.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `0c2cad2d84f0b6284ee0dc2c808a9aaf666b7efa58cffde02cd2aad95112259f`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   TypeError: 'NoneType' object is not subscriptable`
- Red summary: `1 failed, 14 warnings in 3.39s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.55s`

Mutation:

```diff
-            elif paths is not None:
+            elif True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_skips_non_headline_charts_when_paths_are_missing
```

## `test_results_packet_keeps_nonempty_stress_without_a_manifest`

- File: `dashboard/pages/4_Results.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `a6dae3136f97e93a049c4e3c60ed7e93a07d85162fb4cf36cdde5a1c3c896009`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert False`
- Red summary: `1 failed, 14 warnings in 3.64s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.65s`

Mutation:

```diff
-                if stress_delta_df.empty:
+                if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_keeps_nonempty_stress_without_a_manifest
```

## `test_results_packet_drops_empty_stress_and_non_string_previous_run`

- File: `dashboard/pages/4_Results.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `4a0100a5c7f1842e2ebb1862d4291ee2ed88a2b0e81db5c6cc399fa08362d442`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   assert Empty DataFrame\nColumns: []\nIndex: [] is None`
- Red summary: `1 failed, 14 warnings in 3.65s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.66s`

Mutation:

```diff
-                if stress_delta_df.empty:
+                if False:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_drops_empty_stress_and_non_string_previous_run
```

## `test_results_packet_ignores_missing_previous_manifest_path`

- File: `dashboard/pages/4_Results.py`
- Site: `dashboard/pages/4_Results.py:348`, inserted as the `else:` of
  `if candidate.exists():` in `_packet_args` (lines 347-349); this is an
  addition-only mutation with no removed line
- Record: recorded (diff whitespace normalized)
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `cd058d31bd2ce503be28b1b2f1a02916f515dd5ccc50e0eb5f8198737e01c3c0`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   FileNotFoundError: /Users/teacher/.codex/orchestrator/agent-runtime/claude/tmp/pytest-of-teacher/pytest-169/test_results_packet_ignores_mi0/absent-prior.json`
- Red summary: `1 failed, 14 warnings in 3.64s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.66s`

Mutation:

```diff
+                    else:
+                        raise FileNotFoundError(candidate)
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_ignores_missing_previous_manifest_path
```

## `test_results_packet_skips_missing_previous_workbook`

- File: `dashboard/pages/4_Results.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `ae202619b9fabee48cb8e9986d5e1244ab258bf729497ec3221128cef1244fbc`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   TypeError: 'NoneType' object is not subscriptable`
- Red summary: `1 failed, 14 warnings in 3.65s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.65s`

Mutation:

```diff
-                    if prev_out and Path(prev_out).exists():
+                    if True:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_skips_missing_previous_workbook
```

## `test_results_packet_clears_previous_manifest_when_json_is_invalid`

- File: `dashboard/pages/4_Results.py`
- Site: `dashboard/pages/4_Results.py:360`, the `except Exception:` guarding
  the previous-run manifest load in `_packet_args` (not the occurrences at
  lines 160 or 214)
- Record: recorded
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `8ab85fc6580a2884d57565b703722e976b9cb6573a283f7c3fd70fecde639335`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)`
- Red summary: `1 failed, 14 warnings in 3.65s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.70s`

Mutation:

```diff
-                except Exception:
+                except ImportError:
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_clears_previous_manifest_when_json_is_invalid
```

## `test_results_main_reports_packet_export_failures[exc0-Export packet requires Kaleido or Chrome/Chromium-pip install kaleido]`

- File: `dashboard/pages/4_Results.py`
- Record: recorded
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `75327964cbeb1035d680beb7a2e617c12d48b6a9cb73c160d06159f84691f9d5`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   assert False`
- Red summary: `1 failed, 3 deselected, 14 warnings in 3.55s`
- Green exit: 0
- Green signal: `1 passed, 3 deselected, 14 warnings in 3.54s`

Mutation:

```diff
-"📷 Export packet requires Kaleido or Chrome/Chromium for chart generation."
+"📷 Export packet is unavailable."
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_reports_packet_export_failures -k exc0
```

## `test_results_main_reports_packet_export_failures[exc1-Export packet requires Kaleido or Chrome/Chromium-pip install kaleido]`

- File: `dashboard/pages/4_Results.py`
- Record: recorded
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `75327964cbeb1035d680beb7a2e617c12d48b6a9cb73c160d06159f84691f9d5`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   assert False`
- Red summary: `1 failed, 3 deselected, 14 warnings in 3.54s`
- Green exit: 0
- Green signal: `1 passed, 3 deselected, 14 warnings in 3.82s`

Mutation:

```diff
-"📷 Export packet requires Kaleido or Chrome/Chromium for chart generation."
+"📷 Export packet is unavailable."
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_reports_packet_export_failures -k exc1
```

## `test_results_main_reports_packet_export_failures[exc2-Export packet failed: disk full-None]`

- File: `dashboard/pages/4_Results.py`
- Record: recorded
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `ca53945681bde79835630314287f8250af8724a2abdfeef92bd395e5f4d3e829`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   assert False`
- Red summary: `1 failed, 3 deselected, 14 warnings in 3.74s`
- Green exit: 0
- Green signal: `1 passed, 3 deselected, 14 warnings in 3.56s`

Mutation:

```diff
-st.error(f"Export packet failed: {e}")
+st.error(f"Export packet broke: {e}")
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_reports_packet_export_failures -k exc2
```

## `test_results_main_reports_packet_export_failures[exc3-Unexpected error creating export packet: bad packet-individual exports]`

- File: `dashboard/pages/4_Results.py`
- Record: recorded
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `95e3b1d6f491e2503e2156e7a9dcb1ce3a9b688eb729b630ea03bcb929efe911`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   assert False`
- Red summary: `1 failed, 3 deselected, 14 warnings in 3.54s`
- Green exit: 0
- Green signal: `1 passed, 3 deselected, 14 warnings in 3.55s`

Mutation:

```diff
-st.error(f"Unexpected error creating export packet: {e}")
+st.error(f"Export packet surprise: {e}")
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_reports_packet_export_failures -k exc3
```

## `test_results_main_auto_refresh_sleeps_then_reruns`

- File: `dashboard/pages/4_Results.py`
- Record: recorded
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `f11a730e46977cf7b7ae213da69f22a09ed4d79853de8c0f663b5955be5dd57a`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   assert [0] == [60.0]`
- Red summary: `1 failed, 14 warnings in 3.55s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.56s`

Mutation:

```diff
-        time.sleep(interval)
+        time.sleep(0)
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_auto_refresh_sleeps_then_reruns
```

## `test_results_script_entrypoint_invokes_main`

- File: `dashboard/pages/4_Results.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `dc68aee7ee4a1ac6c10a322e4892c6e5e02f7abb28c4a51424d7f8ae27d72677`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   Failed: DID NOT RAISE RuntimeError`
- Red summary: `1 failed, 14 warnings in 3.38s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.43s`

Mutation:

```diff
-    main()
+    pass
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_script_entrypoint_invokes_main
```

## `test_results_main_renders_metrics_plots_and_download_for_available_workbook`

- File: `dashboard/pages/4_Results.py`
- Record: recorded (diff whitespace normalized)
- sha256 before: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9`
- sha256 mutated: `aecd0f5c4aa5c91efc3323503348776c3ba9ad65f3e4b6395cc3e7ec9b1b4044`
- sha256 restored: `868b7ae750bd4a5e1832849f310729ea5a0a753c0609a45eee9fef5c1f9618d9` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert 'Broken PNG' == 'Download PNG'`
- Red summary: `1 failed, 14 warnings in 3.57s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.62s`

Mutation:

```diff
-label="Download PNG",
+label="Broken PNG",
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_renders_metrics_plots_and_download_for_available_workbook
```

## `test_serialize_agent_semantics_from_series`

- File: `pa_core/facade.py`
- Record: recorded (indentation corrected to source)
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `e716af42cbfafdcf340964c0b0931462903ac528ee3f2bc44ab31e4c89676f3a`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert [] == [{'Agent': 'Base', 'n': 2}]`
- Red summary: `1 failed, 14 warnings in 3.37s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.35s`

Mutation:

```diff
-        records = pd.DataFrame([agent_semantics_val]).to_dict(orient="records")
-        inputs["_agent_semantics_df"] = _records_to_builtin(records)
+        inputs["_agent_semantics_df"] = []
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_series
```

## `test_serialize_agent_semantics_from_dataframe_list`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `b596a4e6a12dfd8071865c213fcc23d002dea0049deae0d942b74f8921665acd`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert [] == [{'Agent': 'A...'Agent': 'B'}]`
- Red summary: `1 failed, 14 warnings in 3.32s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.29s`

Mutation:

```diff
-            records = pd.concat(agent_semantics_val, ignore_index=True).to_dict(orient="records")
-            inputs["_agent_semantics_df"] = _records_to_builtin(records)
+            inputs["_agent_semantics_df"] = []
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_dataframe_list
```

## `test_serialize_agent_semantics_from_series_and_dict_list`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `f41c355c331e5c6eaa308df594e6908eb7ec7c5e1321d3e165373a1151a2bc07`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AttributeError: 'dict' object has no attribute 'dtype'`
- Red summary: `1 failed, 14 warnings in 3.31s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.31s`

Mutation:

```diff
-            rows = [
-                item.to_dict() if isinstance(item, pd.Series) else item
-                for item in agent_semantics_val
-            ]
-            records = pd.DataFrame(rows).to_dict(orient="records")
+            records = pd.DataFrame(agent_semantics_val).to_dict(orient="records")
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_series_and_dict_list
```

## `test_serialize_agent_semantics_from_mixed_frame_and_series_list`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `75ae8e8a9cb6fb3e5164a07316a70f855db623c3106e1a48f7fd43fd17d77898`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert [] == [{'Agent': 'A...'Agent': 'B'}]`
- Red summary: `1 failed, 14 warnings in 3.31s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.33s`

Mutation:

```diff
-            frames = []
-            for item in agent_semantics_val:
-                if isinstance(item, pd.DataFrame):
-                    frames.append(item)
-                else:
-                    frames.append(pd.DataFrame([item]))
-            records = pd.concat(frames, ignore_index=True).to_dict(orient="records")
-            inputs["_agent_semantics_df"] = _records_to_builtin(records)
+            inputs["_agent_semantics_df"] = []
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_mixed_frame_and_series_list
```

## `test_serialize_agent_semantics_from_scalar_tuple_and_list (tuple branch)`

- File: `pa_core/facade.py`
- Site: `pa_core/facade.py:133`, the `tuple` branch assignment (the identical
  text at line 135 is the fall-through list assignment and is not mutated)
- Record: replaced: recorded mutation did not fail the original test
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `ef78e1ff66d815864163f14759dd7ec809a63fc7a5750be454d4d5c216b5cfac`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   assert [np.int64(1), np.float64(nan)] == [1, None]`
- Red summary: `1 failed, 14 warnings in 3.29s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.28s`

Mutation:

```diff
-            inputs["_agent_semantics_df"] = _to_builtin_scalar(list(agent_semantics_val))
+            inputs["_agent_semantics_df"] = list(agent_semantics_val)
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_scalar_tuple_and_list
```

## `test_serialize_agent_semantics_from_scalar_tuple_and_list (list branch)`

- File: `pa_core/facade.py`
- Site: `pa_core/facade.py:135`, the fall-through list assignment (the
  identical text at line 133 is the `tuple` branch and is not mutated)
- Record: replaced: recorded mutation did not fail the original test
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `26084457e7c91ebb9223d556f94f772dae80387aad27198963fdc1c4df41899b`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   assert [np.int64(2), inf] == [2, None]`
- Red summary: `1 failed, 14 warnings in 3.28s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.30s`

Mutation:

```diff
-        inputs["_agent_semantics_df"] = _to_builtin_scalar(list(agent_semantics_val))
+        inputs["_agent_semantics_df"] = list(agent_semantics_val)
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_scalar_tuple_and_list
```

## `test_serialize_agent_semantics_from_ndarray`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `9e0f4a75d9e1ac25d3b2e250d651e4442099a118af906f25f254180f4053bf6f`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`
- Red summary: `1 failed, 14 warnings in 3.29s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.32s`

Mutation:

```diff
-        agent_semantics_val = agent_semantics_val.tolist()
+        pass
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_ndarray
```

## `test_serialize_agent_semantics_from_column_mapping`

- File: `pa_core/facade.py`
- Site: `pa_core/facade.py:143`, the `_records_to_builtin(records)` assignment
  in the `dict` branch `try:` block (the other five occurrences of that
  assignment, at lines 93, 97, 108, 118 and 130, are not mutated)
- Record: replaced: recorded mutation targeted the row branch, which this input bypasses
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `8a761db42246a7d2e429ba728297285e4c762997ebd1d12f03b51e457227ac43`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert [{'Agent': 'A... 2, 'w': nan}] == [{'Agent': 'A...2, 'w': None}]`
- Red summary: `1 failed, 14 warnings in 3.31s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.32s`

Mutation:

```diff
-            inputs["_agent_semantics_df"] = _records_to_builtin(records)
+            inputs["_agent_semantics_df"] = records
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_column_mapping
```

## `test_serialize_agent_semantics_keeps_mapping_when_frame_build_fails`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `d31c43fc08b4a24102d74910818e49a4c2b85dad7b46d2164f771a58908c2769`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert None is {'a': [1, 2], 'b': [1]}`
- Red summary: `1 failed, 14 warnings in 3.31s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.32s`

Mutation:

```diff
-            logger.warning(
-                "Failed to serialize _agent_semantics_df mapping for export", exc_info=True
-            )
+            inputs["_agent_semantics_df"] = None
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_keeps_mapping_when_frame_build_fails
```

## `test_serialize_agent_semantics_empty_mapping_becomes_empty_records`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `47cec7c3f1e79582f49c9faaacdff2c02f030f3dff3571c695d18fc70cd45a59`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   assert [{}] == []`
- Red summary: `1 failed, 14 warnings in 3.30s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.30s`

Mutation:

```diff
-        if agent_semantics_val and _mapping_is_row(agent_semantics_val):
+        if _mapping_is_row(agent_semantics_val):
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_empty_mapping_becomes_empty_records
```

## `test_to_builtin_scalar_converts_numpy_and_non_finite_values`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `4f180cb09cc7e7bf8bef9513a7e2043456e7a9abf725574471c38df7bf75dfb8`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   AssertionError: assert nan is None`
- Red summary: `1 failed, 14 warnings in 3.30s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.30s`

Mutation:

```diff
-        return None
+        return value
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_to_builtin_scalar_converts_numpy_and_non_finite_values
```

## `test_export_rejects_unsupported_artifacts_type`

- File: `pa_core/facade.py`
- Record: recorded
- sha256 before: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a`
- sha256 mutated: `59062d0e5fb669a94cf54f352f004f8e0d611022fbb431391431312e990ca86d`
- sha256 restored: `a99e74b13405f332e689342ac718926527cf2ecdbdc064ee474ebae67fe4ea8a` (matches)
- Red exit: 1
- Red signal: `E   Failed: DID NOT RAISE ValueError`
- Red summary: `1 failed, 14 warnings in 3.49s`
- Green exit: 0
- Green signal: `1 passed, 14 warnings in 3.51s`

Mutation:

```diff
-        raise ValueError(f"Unsupported artifacts type: {type(artifacts)}")
+        pass
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_export_rejects_unsupported_artifacts_type
```
