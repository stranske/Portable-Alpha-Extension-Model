# Production-mutation transcripts for issue #2476

Each case edits production bytes, runs the named test, restores the original bytes, and reruns that test.
Test expectations were not changed to force a failure.

## `test_redact_sensitive_ignores_blank_and_non_string_secrets`

- File: `pa_core/llm/result_explain.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert 'prefix [REMOVED] suffix' == 'prefix [REDACTED] suffix'`
- Green signal: `1 passed, 13 warnings in 0.66s`

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
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert '' == 'namespace(content=None)'`
- Green signal: `1 passed, 13 warnings in 0.67s`

Mutation:

```diff
-    return str(response).strip()
-
+    return ""
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_extract_response_text_uses_response_string_when_content_is_not_text
```

## `test_extract_response_text_skips_items_that_are_not_text_or_mappings`

- File: `pa_core/llm/result_explain.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert 'keep 5 tail' == 'keep tail'`
- Green signal: `1 passed, 13 warnings in 0.67s`

Mutation:

```diff
-            if isinstance(item, str):
-                chunks.append(item)
-            elif isinstance(item, Mapping):
-
+            if isinstance(item, str):
+                chunks.append(item)
+            elif not isinstance(item, Mapping):
+                chunks.append(str(item))
+            elif isinstance(item, Mapping):
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_extract_response_text_skips_items_that_are_not_text_or_mappings
```

## `test_extract_metric_delta_ignores_non_mapping_summary_values`

- File: `pa_core/llm/result_explain.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AttributeError: 'str' object has no attribute 'get'`
- Green signal: `1 passed, 13 warnings in 0.66s`

Mutation:

```diff
-    for item in summary.values():
-        if isinstance(item, Mapping):
-            mean = item.get("mean")
-
+    for item in summary.values():
+        if True:
+            mean = item.get("mean")
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_extract_metric_delta_ignores_non_mapping_summary_values
```

## `test_pa_sweep_omits_substream_metadata_when_runner_has_none`

- File: `pa_core/pa.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert {'rng_seed': ...ream_ids': []} == {'rng_seed': None}`
- Green signal: `1 passed, 13 warnings in 0.75s`

Mutation:

```diff
-        if runner.substream_ids:
-            metadata["substream_ids"] = runner.substream_ids
-
+        if True:
+            metadata["substream_ids"] = runner.substream_ids
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_pa_sweep_omits_substream_metadata_when_runner_has_none
```

## `test_run_single_and_sweep_accept_single_column_index_frame`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   ValueError: Index data must be convertible to pandas Series`
- Green signal: `1 passed, 14 warnings in 0.71s`

Mutation:

```diff
-        if not isinstance(idx_series, pd.Series):
-            raise ValueError("Index data must be convertible to pandas Series")
-
+        if True:
+            raise ValueError("Index data must be convertible to pandas Series")
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_run_single_and_sweep_accept_single_column_index_frame
```

## `test_run_single_continues_when_sleeve_validation_reports_no_violations`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   ValueError: Sleeve constraint violations:`
- Green signal: `1 passed, 13 warnings in 0.75s`

Mutation:

```diff
-        if violations:
-            formatted = "\n".join(f"- {violation}" for violation in violations)
-            raise ValueError(f"Sleeve constraint violations:\n{formatted}")
-
+        if True:
+            formatted = "\n".join(f"- {violation}" for violation in violations)
+            raise ValueError(f"Sleeve constraint violations:\n{formatted}")
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_run_single_continues_when_sleeve_validation_reports_no_violations
```

## `test_run_single_records_repair_info_when_correlation_matrices_absent`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert 'broken' == 'absent'`
- Green signal: `1 passed, 13 warnings in 0.69s`

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
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert '_corr_before_df' not in {'backend': 'numpy', 'N_SIMULATIONS': 2, 'N_MONTHS': 3, 'return_unit': 'monthly', ...}`
- Green signal: `1 passed, 13 warnings in 0.69s`

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
- Red exit: 1
- Green exit: 0
- Red signal: `E   AttributeError: 'NoneType' object has no attribute 'get'`
- Green signal: `1 passed, 13 warnings in 0.74s`

Mutation:

```diff
-    elif isinstance(artifacts, SweepArtifacts):
-        metadata = None
-        if artifacts.manifest is not None:
-
+    elif isinstance(artifacts, SweepArtifacts):
+        metadata = None
+        if True:
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_export_sweep_without_manifest_passes_no_metadata
```

## `test_export_normalizes_tuple_agent_semantics_to_a_list`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert <class 'numpy.int64'> is int`
- Green signal: `1 passed, 13 warnings in 0.79s`

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
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert 'broken.xlsx' == 'results.xlsx'`
- Green signal: `1 passed, 13 warnings in 0.69s`

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
- Red exit: 1
- Green exit: 0
- Red signal: `E   json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)`
- Green signal: `1 passed, 13 warnings in 0.76s`

Mutation:

```diff
-        except Exception:
-            manifest_data = None
-
+        except ImportError:
+            manifest_data = None
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_drops_manifest_when_json_cannot_be_parsed
```

## `test_results_main_omits_seed_caption_when_manifest_seed_is_absent`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert ['Seed: None'] == []`
- Green signal: `1 passed, 13 warnings in 0.73s`

Mutation:

```diff
-            seed = manifest_data.get("seed")
-            if seed is not None:
-                st.caption(f"Seed: {seed}")
-
+            seed = manifest_data.get("seed")
+            if True:
+                st.caption(f"Seed: {seed}")
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_omits_seed_caption_when_manifest_seed_is_absent
```

## `test_results_main_omits_tracking_error_when_neither_column_is_present`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   assert False`
- Green signal: `1 passed, 13 warnings in 0.75s`

Mutation:

```diff
-    if SUMMARY_TE_COLUMN in summary:
-        te_column = SUMMARY_TE_COLUMN
-
+    if True:
+        te_column = SUMMARY_CVAR_COLUMN
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_omits_tracking_error_when_neither_column_is_present
```

## `test_results_main_skips_non_headline_charts_when_paths_are_missing`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   TypeError: 'NoneType' object is not subscriptable`
- Green signal: `1 passed, 13 warnings in 0.74s`

Mutation:

```diff
-            elif paths is not None:
-                sub_paths = paths[agents].iloc[:, :months]
-
+            elif True:
+                sub_paths = paths[agents].iloc[:, :months]
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_skips_non_headline_charts_when_paths_are_missing
```

## `test_results_packet_keeps_nonempty_stress_without_a_manifest`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert False`
- Green signal: `1 passed, 13 warnings in 0.77s`

Mutation:

```diff
-                if stress_delta_df.empty:
-                    stress_delta_df = None
-
+                if True:
+                    stress_delta_df = None
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_keeps_nonempty_stress_without_a_manifest
```

## `test_results_packet_drops_empty_stress_and_non_string_previous_run`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   assert Empty DataFrame\nColumns: []\nIndex: [] is None`
- Green signal: `1 passed, 13 warnings in 0.76s`

Mutation:

```diff
-                if stress_delta_df.empty:
-                    stress_delta_df = None
-
+                if False:
+                    stress_delta_df = None
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_drops_empty_stress_and_non_string_previous_run
```

## `test_results_packet_ignores_missing_previous_manifest_path`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   FileNotFoundError: /tmp/pytest-of-runner/pytest-26/test_results_packet_ignores_mi0/absent-prior.json`
- Green signal: `1 passed, 13 warnings in 0.77s`

Mutation:

```diff
-                    if candidate.exists():
-                        prev_manifest_path = candidate
-
+                    if candidate.exists():
+                        prev_manifest_path = candidate
+                    else:
+                        raise FileNotFoundError(candidate)
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_ignores_missing_previous_manifest_path
```

## `test_results_packet_skips_missing_previous_workbook`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   TypeError: 'NoneType' object is not subscriptable`
- Green signal: `1 passed, 13 warnings in 0.77s`

Mutation:

```diff
-                    if prev_out and Path(prev_out).exists():
-                        prev_summary_df = pd.read_excel(prev_out, sheet_name=SUMMARY_SHEET_NAME)
-
+                    if True:
+                        prev_summary_df = pd.read_excel(prev_out, sheet_name=SUMMARY_SHEET_NAME)
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_skips_missing_previous_workbook
```

## `test_results_packet_clears_previous_manifest_when_json_is_invalid`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)`
- Green signal: `1 passed, 13 warnings in 0.76s`

Mutation:

```diff
-                except Exception:
-                    prev_manifest_data = None
-                    prev_summary_df = None
-
+                except ImportError:
+                    prev_manifest_data = None
+                    prev_summary_df = None
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_packet_clears_previous_manifest_when_json_is_invalid
```

## `test_results_main_reports_packet_export_failures[exc0-Export packet requires Kaleido or Chrome/Chromium-pip install kaleido]`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   assert False`
- Green signal: `1 passed, 13 warnings in 0.76s`

Mutation:

```diff
-"📷 Export packet requires Kaleido or Chrome/Chromium for chart generation."
+"📷 Export packet is unavailable."
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_reports_packet_export_failures[exc0-Export packet requires Kaleido or Chrome/Chromium-pip install kaleido]
```

## `test_results_main_reports_packet_export_failures[exc1-Export packet requires Kaleido or Chrome/Chromium-pip install kaleido]`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   assert False`
- Green signal: `1 passed, 13 warnings in 0.74s`

Mutation:

```diff
-"📷 Export packet requires Kaleido or Chrome/Chromium for chart generation."
+"📷 Export packet is unavailable."
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_reports_packet_export_failures[exc1-Export packet requires Kaleido or Chrome/Chromium-pip install kaleido]
```

## `test_results_main_reports_packet_export_failures[exc2-Export packet failed: disk full-None]`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   assert False`
- Green signal: `1 passed, 13 warnings in 0.74s`

Mutation:

```diff
-st.error(f"Export packet failed: {e}")
+st.error(f"Export packet broke: {e}")
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_reports_packet_export_failures[exc2-Export packet failed: disk full-None]
```

## `test_results_main_reports_packet_export_failures[exc3-Unexpected error creating export packet: bad packet-individual exports]`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   assert False`
- Green signal: `1 passed, 13 warnings in 0.74s`

Mutation:

```diff
-st.error(f"Unexpected error creating export packet: {e}")
+st.error(f"Export packet surprise: {e}")
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_reports_packet_export_failures[exc3-Unexpected error creating export packet: bad packet-individual exports]
```

## `test_results_main_auto_refresh_sleeps_then_reruns`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   assert [0] == [60.0]`
- Green signal: `1 passed, 13 warnings in 0.73s`

Mutation:

```diff
-        time.sleep(interval)
-        st.rerun()
-
+        time.sleep(0)
+        st.rerun()
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_auto_refresh_sleeps_then_reruns
```

## `test_results_script_entrypoint_invokes_main`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   Failed: DID NOT RAISE RuntimeError`
- Green signal: `1 passed, 13 warnings in 0.68s`

Mutation:

```diff
-if __name__ == "__main__":
-    main()
-
+if __name__ == "__main__":
+    pass
+
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_script_entrypoint_invokes_main
```

## `test_results_main_renders_metrics_plots_and_download_for_available_workbook`

- File: `dashboard/pages/4_Results.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert 'Broken PNG' == 'Download PNG'`
- Green signal: `1 passed, 13 warnings in 0.73s`

Mutation:

```diff
-label="Download PNG"
+label="Broken PNG"
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2381.py::test_results_main_renders_metrics_plots_and_download_for_available_workbook
```

## `test_serialize_agent_semantics_from_series`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert [] == [{'Agent': 'Base', 'n': 2}]`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-        if isinstance(agent_semantics_val, pd.Series):
-            records = pd.DataFrame([agent_semantics_val]).to_dict(orient="records")
-            inputs["_agent_semantics_df"] = _records_to_builtin(records)
-            return
+        if isinstance(agent_semantics_val, pd.Series):
+            inputs["_agent_semantics_df"] = []
+            return
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_series
```

## `test_serialize_agent_semantics_from_dataframe_list`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert [] == [{'Agent': 'A'}, {'Agent': 'B'}]`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-        if agent_semantics_val and all(
-            isinstance(item, pd.DataFrame) for item in agent_semantics_val
-        ):
-            records = pd.concat(agent_semantics_val, ignore_index=True).to_dict(orient="records")
-            inputs["_agent_semantics_df"] = _records_to_builtin(records)
-            return
+        if agent_semantics_val and all(
+            isinstance(item, pd.DataFrame) for item in agent_semantics_val
+        ):
+            inputs["_agent_semantics_df"] = []
+            return
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_dataframe_list
```

## `test_serialize_agent_semantics_from_series_and_dict_list`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AttributeError: 'dict' object has no attribute 'dtype'`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-        if agent_semantics_val and all(
-            isinstance(item, (pd.Series, dict)) for item in agent_semantics_val
-        ):
-            rows = [
-                item.to_dict() if isinstance(item, pd.Series) else item
-                for item in agent_semantics_val
-            ]
-            records = pd.DataFrame(rows).to_dict(orient="records")
-            inputs["_agent_semantics_df"] = _records_to_builtin(records)
-            return
+        if agent_semantics_val and all(
+            isinstance(item, (pd.Series, dict)) for item in agent_semantics_val
+        ):
+            records = pd.DataFrame(agent_semantics_val).to_dict(orient="records")
+            inputs["_agent_semantics_df"] = _records_to_builtin(records)
+            return
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_series_and_dict_list
```

## `test_serialize_agent_semantics_from_mixed_frame_and_series_list`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert [] == [{'Agent': 'A'}, {'Agent': 'B'}]`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-        if agent_semantics_val and all(
-            isinstance(item, (pd.DataFrame, pd.Series, dict)) for item in agent_semantics_val
-        ):
-            frames = []
-            for item in agent_semantics_val:
-                if isinstance(item, pd.DataFrame):
-                    frames.append(item)
-                else:
-                    frames.append(pd.DataFrame([item]))
-            records = pd.concat(frames, ignore_index=True).to_dict(orient="records")
-            inputs["_agent_semantics_df"] = _records_to_builtin(records)
-            return
+        if agent_semantics_val and all(
+            isinstance(item, (pd.DataFrame, pd.Series, dict)) for item in agent_semantics_val
+        ):
+            inputs["_agent_semantics_df"] = []
+            return
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_mixed_frame_and_series_list
```

## `test_serialize_agent_semantics_from_scalar_tuple_and_list`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert [1, 2] == [1, 2]`
- Green signal: `3 passed, 2 warnings in 0.70s`

Mutation:

```diff
-        inputs["_agent_semantics_df"] = _to_builtin_scalar(list(agent_semantics_val))
-        return
+        inputs["_agent_semantics_df"] = list(agent_semantics_val)
+        return
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_scalar_tuple_and_list
```

## `test_serialize_agent_semantics_from_ndarray`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert '[[1 2]...' == [[1, 2], [3, 4]]`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-    if np_module is not None and isinstance(agent_semantics_val, np_module.ndarray):
-        agent_semantics_val = agent_semantics_val.tolist()
+    if np_module is not None and isinstance(agent_semantics_val, np_module.ndarray):
+        pass
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_ndarray
```

## `test_serialize_agent_semantics_from_column_mapping`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert [{'Agent...'n': 1}] == [{'Agent': 'A', 'n': 1}, {'Agent': 'B', 'n': 2}]`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-        if agent_semantics_val and _mapping_is_row(agent_semantics_val):
-            inputs["_agent_semantics_df"] = _records_to_builtin([agent_semantics_val])
-            return
+        if agent_semantics_val and _mapping_is_row(agent_semantics_val):
+            return
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_from_column_mapping
```

## `test_serialize_agent_semantics_keeps_mapping_when_frame_build_fails`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert None == {'a': [1, 2], 'b': [1]}`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-        try:
-            records = pd.DataFrame(agent_semantics_val).to_dict(orient="records")
-            inputs["_agent_semantics_df"] = _records_to_builtin(records)
-        except (ValueError, TypeError):
-            logger.warning(
-                "Failed to serialize _agent_semantics_df mapping for export", exc_info=True
-            )
-            return
+        try:
+            records = pd.DataFrame(agent_semantics_val).to_dict(orient="records")
+            inputs["_agent_semantics_df"] = _records_to_builtin(records)
+        except (ValueError, TypeError):
+            inputs["_agent_semantics_df"] = None
+            return
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_keeps_mapping_when_frame_build_fails
```

## `test_serialize_agent_semantics_empty_mapping_becomes_empty_records`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert {} == []`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-        if agent_semantics_val and _mapping_is_row(agent_semantics_val):
-            inputs["_agent_semantics_df"] = _records_to_builtin([agent_semantics_val])
-            return
+        if _mapping_is_row(agent_semantics_val):
+            inputs["_agent_semantics_df"] = _records_to_builtin([agent_semantics_val])
+            return
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_serialize_agent_semantics_empty_mapping_becomes_empty_records
```

## `test_to_builtin_scalar_converts_numpy_and_non_finite_values`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   AssertionError: assert <NAN> is None`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-    if isinstance(value, float) and not math.isfinite(value):
-        return None
+    if isinstance(value, float) and not math.isfinite(value):
+        return value
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_to_builtin_scalar_converts_numpy_and_non_finite_values
```

## `test_export_rejects_unsupported_artifacts_type`

- File: `pa_core/facade.py`
- Red exit: 1
- Green exit: 0
- Red signal: `E   Failed: DID NOT RAISE ValueError`
- Green signal: `1 passed, 2 warnings in 0.70s`

Mutation:

```diff
-    else:
-        raise ValueError(f"Unsupported artifacts type: {type(artifacts)}")
+    else:
+        pass
```

Command:

```sh
python -m pytest -q --tb=line -p no:cacheprovider tests/test_coverage_round_2476_facade.py::test_export_rejects_unsupported_artifacts_type
```
