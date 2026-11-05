# Mutation transcripts

Check date: 2026-11-03 America/Chicago.
Interpreter: /opt/anaconda3/bin/python3; Python 3.12.2. Production baseline is main 7613ffd9
plus the one-token attribution fix (np -> _np). Historical evidence is retained separately.

## test_compare_run_helpers_reject_unusable_artifacts

Production mutation: `pa_core/llm/compare_runs.py`

```diff
-    if not isinstance(loaded, dict):
-        return None, prev_manifest_path
+    if not isinstance(loaded, dict):
+        return {}, prev_manifest_path
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts
F                                                                        [100%]
=================================== FAILURES ===================================
______________ test_compare_run_helpers_reject_unusable_artifacts ______________
tests/test_coverage_round_2375.py:36: in test_compare_run_helpers_reject_unusable_artifacts
    assert load_prior_manifest({"previous_run": str(non_mapping_path)}) == (
E   AssertionError: assert ({}, PosixPat...nifest.json')) == (None, PosixP...nifest.json'))
E     
E     At index 0 diff: {} != None
E     Use -v to get more diff
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts
1 failed, 2 warnings in 2.44s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts  # restored
.                                                                        [100%]
1 passed, 2 warnings in 1.64s
exit=0
```

## test_compare_runs_explains_missing_prior_summary

Production mutation: `pa_core/llm/compare_runs.py`

```diff
-raise ValueError(f"No readable prior summary found at expected path: {expected}")
+pass  # mutation
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_compare_runs_explains_missing_prior_summary
F                                                                        [100%]
=================================== FAILURES ===================================
_______________ test_compare_runs_explains_missing_prior_summary _______________
tests/test_coverage_round_2375.py:50: in test_compare_runs_explains_missing_prior_summary
    compare_runs(
pa_core/llm/compare_runs.py:406: in compare_runs
    metrics_b = build_metric_catalog(prior_summary)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pa_core/llm/compare_runs.py:295: in build_metric_catalog
    raise TypeError("summary_df must be a pandas DataFrame")
E   TypeError: summary_df must be a pandas DataFrame
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_compare_runs_explains_missing_prior_summary
1 failed, 2 warnings in 1.81s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_compare_runs_explains_missing_prior_summary  # restored
.                                                                        [100%]
1 passed, 2 warnings in 1.61s
exit=0
```

## test_compare_runs_rejects_missing_prior_manifest

Production mutation: `pa_core/llm/compare_runs.py`

```diff
-raise ValueError("No readable prior manifest found from manifest_data['previous_run'].")
+pass  # mutation
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_compare_runs_rejects_missing_prior_manifest
F                                                                        [100%]
=================================== FAILURES ===================================
_______________ test_compare_runs_rejects_missing_prior_manifest _______________
tests/test_coverage_round_2375.py:59: in test_compare_runs_rejects_missing_prior_manifest
    with pytest.raises(ValueError, match="No readable prior manifest"):
E   AssertionError: Regex pattern did not match.
E     Expected regex: 'No readable prior manifest'
E     Actual message: 'No readable prior summary found at expected path: <unset>'
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_compare_runs_rejects_missing_prior_manifest
1 failed, 2 warnings in 1.79s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_compare_runs_rejects_missing_prior_manifest  # restored
.                                                                        [100%]
1 passed, 2 warnings in 1.65s
exit=0
```

## test_compare_runs_survives_fleet_event_failure

Production mutation: `pa_core/llm/compare_runs.py`

```diff
-    except Exception:
-        logger.warning("Failed to record run-comparison fleet event", exc_info=True)
+    except Exception:
+        raise
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_compare_runs_survives_fleet_event_failure
F                                                                        [100%]
=================================== FAILURES ===================================
________________ test_compare_runs_survives_fleet_event_failure ________________
tests/test_coverage_round_2375.py:87: in test_compare_runs_survives_fleet_event_failure
    text, _, payload = compare_runs(
pa_core/llm/compare_runs.py:476: in compare_runs
    record_fleet_event(
tests/test_coverage_round_2375.py:84: in <lambda>
    lambda _event: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_coverage_round_2375.py:84: in <genexpr>
    lambda _event: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
E   RuntimeError: telemetry unavailable
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_compare_runs_survives_fleet_event_failure
1 failed, 2 warnings in 1.77s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_compare_runs_survives_fleet_event_failure  # restored
.                                                                        [100%]
1 passed, 2 warnings in 1.70s
exit=0
```

## test_metric_catalog_rejects_non_dataframe_and_empty_delta

Production mutation: `pa_core/llm/compare_runs.py`

```diff
-        raise TypeError("summary_df must be a pandas DataFrame")
+        return {}
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta
F                                                                        [100%]
=================================== FAILURES ===================================
__________ test_metric_catalog_rejects_non_dataframe_and_empty_delta ___________
tests/test_coverage_round_2375.py:99: in test_metric_catalog_rejects_non_dataframe_and_empty_delta
    with pytest.raises(TypeError, match="pandas DataFrame"):
E   Failed: DID NOT RAISE TypeError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta
1 failed, 2 warnings in 1.75s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta  # restored
.                                                                        [100%]
1 passed, 2 warnings in 1.58s
exit=0
```

## test_single_run_packet_failure (RuntimeError)

Production mutation: `pa_core/cli.py`

```diff
-                print(f"   📋 PowerPoint: {pptx_path}")
-            except RuntimeError as e:
-                print(f"❌ Export packet failed: {e}")
-                _emit_run_end()
+                print(f"   📋 PowerPoint: {pptx_path}")
+            except RuntimeError as e:
+                print(f"❌ Export packet failed (mutated): {e}")
+                _emit_run_end()
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]'
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable] _
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:186: in test_single_run_packet_failure_reports_the_failure_class
    assert f"❌ {expected}" in capsys.readouterr().out
E   AssertionError: assert '❌ Export packet failed: packet unavailable' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n❌ Export packet failed (mutated): packet unavailable\n'
E    +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n❌ Export packet failed (mutated): packet unavailable\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...nsitivity analysis completed. Evaluated 12 scenarios.\n❌ Export packet failed (mutated): packet unavailable\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...nsitivity analysis completed. Evaluated 12 scenarios.\n❌ Export packet failed (mutated): packet unavailable\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x3014768a0>.readouterr
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]
1 failed, 17 warnings in 2.21s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.10s
exit=0
```

## test_single_run_packet_failure (ImportError)

Production mutation: `pa_core/cli.py`

```diff
-                return
-            except (ImportError, ModuleNotFoundError) as e:
-                logger.error(f"Export packet failed due to missing dependency: {e}")
-                print(f"❌ Export packet failed due to missing dependency: {e}")
-                print(
-                    "💡 Install required packages: pip install plotly kaleido openpyxl python-pptx"
-                )
+                return
+            except (ImportError, ModuleNotFoundError) as e:
+                logger.error(f"Export packet failed due to missing dependency: {e}")
+                print(f"❌ Export packet failed due to missing dependency (mutated): {e}")
+                print(
+                    "💡 Install required packages: pip install plotly kaleido openpyxl python-pptx"
+                )
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]'
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable] _
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:186: in test_single_run_packet_failure_reports_the_failure_class
    assert f"❌ {expected}" in capsys.readouterr().out
E   AssertionError: assert '❌ Export packet failed due to missing dependency: packet unavailable' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...pendency (mutated): packet unavailable\n💡 Install required packages: pip install plotly kaleido openpyxl python-pptx\n'
E    +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...pendency (mutated): packet unavailable\n💡 Install required packages: pip install plotly kaleido openpyxl python-pptx\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...(mutated): packet unavailable\n💡 Install required packages: pip install plotly kaleido openpyxl python-pptx\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...(mutated): packet unavailable\n💡 Install required packages: pip install plotly kaleido openpyxl python-pptx\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x148a76c60>.readouterr
------------------------------ Captured log call -------------------------------
ERROR    pa_core.cli:cli.py:2024 Export packet failed due to missing dependency: packet unavailable
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]
1 failed, 17 warnings in 2.17s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.02s
exit=0
```

## test_single_run_packet_failure (ValueError)

Production mutation: `pa_core/cli.py`

```diff
-            except (ValueError, TypeError, KeyError) as e:
-                logger.error(f"Export packet failed due to data/config issue: {e}")
-                print(f"❌ Export packet failed due to data or configuration issue: {e}")
+            except (ValueError, TypeError, KeyError) as e:
+                logger.error(f"Export packet failed due to data/config issue: {e}")
+                print(f"❌ Export packet failed due to data or configuration issue (mutated): {e}")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]'
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable] _
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:186: in test_single_run_packet_failure_reports_the_failure_class
    assert f"❌ {expected}" in capsys.readouterr().out
E   AssertionError: assert '❌ Export packet failed due to data or configuration issue: packet unavailable' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ue to data or configuration issue (mutated): packet unavailable\n💡 Check your data inputs and configuration settings\n'
E    +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ue to data or configuration issue (mutated): packet unavailable\n💡 Check your data inputs and configuration settings\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...a or configuration issue (mutated): packet unavailable\n💡 Check your data inputs and configuration settings\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...a or configuration issue (mutated): packet unavailable\n💡 Check your data inputs and configuration settings\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x158176930>.readouterr
------------------------------ Captured log call -------------------------------
ERROR    pa_core.cli:cli.py:2032 Export packet failed due to data/config issue: packet unavailable
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]
1 failed, 17 warnings in 2.14s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 1.98s
exit=0
```

## test_single_run_packet_failure (OSError)

Production mutation: `pa_core/cli.py`

```diff
-            except (OSError, PermissionError) as e:
-                logger.error(f"Export packet failed due to file system issue: {e}")
-                print(f"❌ Export packet failed due to file system issue: {e}")
+            except (OSError, PermissionError) as e:
+                logger.error(f"Export packet failed due to file system issue: {e}")
+                print(f"❌ Export packet failed due to file system issue (mutated): {e}")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]'
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable] _
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:186: in test_single_run_packet_failure_reports_the_failure_class
    assert f"❌ {expected}" in capsys.readouterr().out
E   AssertionError: assert '❌ Export packet failed due to file system issue: packet unavailable' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ket failed due to file system issue (mutated): packet unavailable\n💡 Check file permissions and available disk space\n'
E    +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ket failed due to file system issue (mutated): packet unavailable\n💡 Check file permissions and available disk space\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...d due to file system issue (mutated): packet unavailable\n💡 Check file permissions and available disk space\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...d due to file system issue (mutated): packet unavailable\n💡 Check file permissions and available disk space\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x146b76cc0>.readouterr
------------------------------ Captured log call -------------------------------
ERROR    pa_core.cli:cli.py:2038 Export packet failed due to file system issue: packet unavailable
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]
1 failed, 17 warnings in 2.11s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.01s
exit=0
```

## test_single_run_gif_conversion_error_is_reported

Production mutation: `pa_core/cli.py`

```diff
-            except (ValueError, TypeError) as e:
-                print(f"❌ GIF export failed: Data conversion error - {e}")
+            except (ValueError, TypeError) as e:
+                print(f"❌ GIF export failed: Data conversion error (mutated) - {e}")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported
F                                                                        [100%]
=================================== FAILURES ===================================
_______________ test_single_run_gif_conversion_error_is_reported _______________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:205: in test_single_run_gif_conversion_error_is_reported
    assert "❌ GIF export failed: Data conversion error - not numeric" in capsys.readouterr().out
E   AssertionError: assert '❌ GIF export failed: Data conversion error - not numeric' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============... export failed: Data conversion error (mutated) - not numeric\n💡 Check that return data contains only numeric values\n'
E    +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============... export failed: Data conversion error (mutated) - not numeric\n💡 Check that return data contains only numeric values\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...ailed: Data conversion error (mutated) - not numeric\n💡 Check that return data contains only numeric values\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...ailed: Data conversion error (mutated) - not numeric\n💡 Check that return data contains only numeric values\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x177776ae0>.readouterr
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported
1 failed, 17 warnings in 2.09s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.09s
exit=0
```

## test_single_run_dashboard_missing_file_is_reported

Production mutation: `pa_core/cli.py`

```diff
-            except FileNotFoundError as e:
-                print(f"❌ Dashboard launch failed: {e}")
+            except FileNotFoundError as e:
+                print(f"❌ Dashboard launch failed (mutated): {e}")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported
F                                                                        [100%]
=================================== FAILURES ===================================
______________ test_single_run_dashboard_missing_file_is_reported ______________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:218: in test_single_run_dashboard_missing_file_is_reported
    assert "❌ Dashboard launch failed: Dashboard file not found" in capsys.readouterr().out
E   assert '❌ Dashboard launch failed: Dashboard file not found' in "[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============... Dashboard file not found: dashboard/app.py\n💡 Ensure the dashboard files are present in the 'dashboard/' directory.\n"
E    +  where "[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============... Dashboard file not found: dashboard/app.py\n💡 Ensure the dashboard files are present in the 'dashboard/' directory.\n" = CaptureResult(out="[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...d file not found: dashboard/app.py\n💡 Ensure the dashboard files are present in the 'dashboard/' directory.\n", err='').out
E    +    where CaptureResult(out="[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...d file not found: dashboard/app.py\n💡 Ensure the dashboard files are present in the 'dashboard/' directory.\n", err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x304e76f00>.readouterr
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported
1 failed, 17 warnings in 2.08s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported  # restored
.                                                                        [100%]
1 passed, 17 warnings in 1.97s
exit=0
```

## test_single_run_bundle_contains_the_completed_artifacts

Production mutation: `pa_core/cli.py`

```diff
-    _record_artifact(flags.save_xlsx or "Outputs.xlsx")
+    pass  # omitted completed workbook registration
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts
F                                                                        [100%]
=================================== FAILURES ===================================
___________ test_single_run_bundle_contains_the_completed_artifacts ____________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:242: in test_single_run_bundle_contains_the_completed_artifacts
    assert isinstance(outputs_meta, dict) and outputs_meta
E   assert (True and {})
E    +  where True = isinstance({}, dict)
----------------------------- Captured stdout call -----------------------------
[BACKEND] Using backend: numpy

🔍 Running sensitivity analysis...

📊 Sensitivity Analysis Results:
==================================================
📈 sigma_E_+5%          | Delta:  +0.0037%
📈 sigma_M_-5%          | Delta:  +0.0036%
📉 sigma_E_-5%          | Delta:  -0.0009%
📈 mu_H_+5%             | Delta:  +0.0008%
📉 mu_H_-5%             | Delta:  -0.0008%
📉 sigma_M_+5%          | Delta:  -0.0007%
📉 sigma_H_-5%          | Delta:  -0.0004%
📈 sigma_H_+5%          | Delta:  +0.0004%
📉 mu_E_+5%             | Delta:  +0.0000%
📉 mu_E_-5%             | Delta:  +0.0000%
📉 mu_M_+5%             | Delta:  +0.0000%
📉 mu_M_-5%             | Delta:  +0.0000%

✅ Sensitivity analysis completed. Evaluated 12 scenarios.
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts
1 failed, 17 warnings in 2.06s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts  # restored
.                                                                        [100%]
1 passed, 17 warnings in 1.93s
exit=0
```

## test_single_run_rejects_daily_index_before_simulation

Production mutation: `pa_core/cli.py`

```diff
-        raise SystemExit(f"Error: {e}") from None
+        raise ValueError(f"Error: {e}") from None
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_rejects_daily_index_before_simulation
F                                                                        [100%]
=================================== FAILURES ===================================
____________ test_single_run_rejects_daily_index_before_simulation _____________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:268: in test_single_run_rejects_daily_index_before_simulation
    main(_single_run_argv(config, index, output))
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/pa_core/cli.py:399: in main
    _main(argv, deps, warning_collector=warning_collector)
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/pa_core/cli.py:1173: in _main
    raise ValueError(f"Error: {e}") from None
E   ValueError: Error: Expected monthly data, got daily. Use --resample monthly to convert.
----------------------------- Captured stdout call -----------------------------
[BACKEND] Using backend: numpy
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_rejects_daily_index_before_simulation
1 failed, 4 warnings in 2.16s

exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_rejects_daily_index_before_simulation # restored
.                                                                        [100%]
1 passed, 4 warnings in 1.79s

exit=0
```

## test_single_run_dashboard_launch_failure[exit-code]

Production mutation: `pa_core/cli.py`

```diff
-print(f"❌ Dashboard launch failed with exit code {e.returncode}")
+print("mutated dashboard diagnosis")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[exit-code]'
F                                                                        [100%]
=================================== FAILURES ===================================
_____________ test_single_run_dashboard_launch_failure[exit-code] ______________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:300: in test_single_run_dashboard_launch_failure
    assert (
E   assert '❌ Dashboard launch failed with exit code 7' in "[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...t\n   • Check if 'dashboard/app.py' is valid Python code\n   • Verify your Python environment is properly configured\n"
E    +  where "[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...t\n   • Check if 'dashboard/app.py' is valid Python code\n   • Verify your Python environment is properly configured\n" = CaptureResult(out="[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...heck if 'dashboard/app.py' is valid Python code\n   • Verify your Python environment is properly configured\n", err='').out
E    +    where CaptureResult(out="[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...heck if 'dashboard/app.py' is valid Python code\n   • Verify your Python environment is properly configured\n", err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x15ca76600>.readouterr
------------------------------ Captured log call -------------------------------
ERROR    pa_core.cli:cli.py:2186 Dashboard launch failed with exit code 7: Command '['streamlit']' returned non-zero exit status 7.
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[exit-code]
1 failed, 17 warnings in 2.11s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[exit-code]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 1.95s
exit=0
```

## test_single_run_dashboard_launch_failure[dependency]

Production mutation: `pa_core/cli.py`

```diff
-print(f"❌ Dashboard launch failed: Streamlit not available - {e}")
+print("mutated dashboard diagnosis")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[dependency]'
F                                                                        [100%]
=================================== FAILURES ===================================
_____________ test_single_run_dashboard_launch_failure[dependency] _____________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:300: in test_single_run_dashboard_launch_failure
    assert (
E   AssertionError: assert '❌ Dashboard launch failed: Streamlit not available - streamlit missing' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...analysis completed. Evaluated 12 scenarios.\nmutated dashboard diagnosis\n💡 Install Streamlit: pip install streamlit\n'
E    +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...analysis completed. Evaluated 12 scenarios.\nmutated dashboard diagnosis\n💡 Install Streamlit: pip install streamlit\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...completed. Evaluated 12 scenarios.\nmutated dashboard diagnosis\n💡 Install Streamlit: pip install streamlit\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...completed. Evaluated 12 scenarios.\nmutated dashboard diagnosis\n💡 Install Streamlit: pip install streamlit\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x309553b30>.readouterr
------------------------------ Captured log call -------------------------------
ERROR    pa_core.cli:cli.py:2195 Dashboard launch failed due to missing streamlit: streamlit missing
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[dependency]
1 failed, 17 warnings in 2.05s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[dependency]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.01s
exit=0
```

## test_single_run_dashboard_launch_failure[permission]

Production mutation: `pa_core/cli.py`

```diff
-print(f"❌ Dashboard launch failed: System/permission error - {e}")
+print("mutated dashboard diagnosis")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[permission]'
F                                                                        [100%]
=================================== FAILURES ===================================
_____________ test_single_run_dashboard_launch_failure[permission] _____________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:300: in test_single_run_dashboard_launch_failure
    assert (
E   AssertionError: assert '❌ Dashboard launch failed: System/permission error - launch denied' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...lysis completed. Evaluated 12 scenarios.\nmutated dashboard diagnosis\n💡 Check file permissions and system resources\n'
E    +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...lysis completed. Evaluated 12 scenarios.\nmutated dashboard diagnosis\n💡 Check file permissions and system resources\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...pleted. Evaluated 12 scenarios.\nmutated dashboard diagnosis\n💡 Check file permissions and system resources\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...pleted. Evaluated 12 scenarios.\nmutated dashboard diagnosis\n💡 Check file permissions and system resources\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x176d54ec0>.readouterr
------------------------------ Captured log call -------------------------------
ERROR    pa_core.cli:cli.py:2201 Dashboard launch failed due to system issue: launch denied
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[permission]
1 failed, 17 warnings in 2.11s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_dashboard_launch_failure[permission]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.19s
exit=0
```

## test_single_run_gif_export_failure[renderer]

Production mutation: `pa_core/cli.py`

```diff
-                    print("❌ GIF export failed: Chrome/Chromium required")
+                    print("mutated renderer diagnosis")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_gif_export_failure[renderer]'
F                                                                        [100%]
=================================== FAILURES ===================================
_________________ test_single_run_gif_export_failure[renderer] _________________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:335: in test_single_run_gif_export_failure
    assert f"❌ GIF export failed: {expected}" in capsys.readouterr().out
E   AssertionError: assert '❌ GIF export failed: Chrome/Chromium required' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ompleted. Evaluated 12 scenarios.\nmutated renderer diagnosis\n💡 Install with: sudo apt-get install chromium-browser\n'
E    +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ompleted. Evaluated 12 scenarios.\nmutated renderer diagnosis\n💡 Install with: sudo apt-get install chromium-browser\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul... Evaluated 12 scenarios.\nmutated renderer diagnosis\n💡 Install with: sudo apt-get install chromium-browser\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul... Evaluated 12 scenarios.\nmutated renderer diagnosis\n💡 Install with: sudo apt-get install chromium-browser\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x308576810>.readouterr
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_gif_export_failure[renderer]
1 failed, 17 warnings in 2.42s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_gif_export_failure[renderer]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 1.98s
exit=0
```

## test_single_run_gif_export_failure[encoder]

Production mutation: `pa_core/cli.py`

```diff
-                    print(f"❌ GIF export failed: {e}")
+                    print("mutated encoder diagnosis")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_gif_export_failure[encoder]'
F                                                                        [100%]
=================================== FAILURES ===================================
_________________ test_single_run_gif_export_failure[encoder] __________________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:335: in test_single_run_gif_export_failure
    assert f"❌ GIF export failed: {expected}" in capsys.readouterr().out
E   AssertionError: assert '❌ GIF export failed: encoder failed' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...          | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\nmutated encoder diagnosis\n'
E    +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...          | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\nmutated encoder diagnosis\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul... | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\nmutated encoder diagnosis\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul... | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\nmutated encoder diagnosis\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x15f9751f0>.readouterr
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_gif_export_failure[encoder]
1 failed, 17 warnings in 2.12s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_gif_export_failure[encoder]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.06s
exit=0
```

## test_single_run_attribution_fallback_exports_totals

Production mutation: `pa_core/cli.py`

```diff
-                mean_month = float(_np.asarray(arr).mean())
+                mean_month = float(np.asarray(arr).mean())
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_attribution_fallback_exports_totals
F                                                                        [100%]
=================================== FAILURES ===================================
_____________ test_single_run_attribution_fallback_exports_totals ______________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:346: in unavailable
    raise TypeError("detailed attribution unavailable")
E   TypeError: detailed attribution unavailable

During handling of the above exception, another exception occurred:
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:351: in test_single_run_attribution_fallback_exports_totals
    main(_single_run_argv(config, index, output))
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/pa_core/cli.py:399: in main
    _main(argv, deps, warning_collector=warning_collector)
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/pa_core/cli.py:1555: in _main
    mean_month = float(np.asarray(arr).mean())
                       ^^
E   NameError: name 'np' is not defined
----------------------------- Captured stdout call -----------------------------
[BACKEND] Using backend: numpy
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_attribution_fallback_exports_totals
1 failed, 5 warnings in 2.11s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings tests/test_coverage_round_2375.py::test_single_run_attribution_fallback_exports_totals  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.20s
exit=0
```

## test_single_run_tradeoff_table[export]

Production mutation: `pa_core/cli.py`

```diff
-trade_df.head(max(1, args.tradeoff_top)).reset_index(
+trade_df.reset_index(
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_tradeoff_table[export]'
F                                                                        [100%]
=================================== FAILURES ===================================
____________________ test_single_run_tradeoff_table[export] ____________________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:384: in test_single_run_tradeoff_table
    pd.testing.assert_frame_equal(actual, suggestions.head(1), check_dtype=False)
E   AssertionError: DataFrame are different
E   
E   DataFrame shape mismatch
E   [left]:  (2, 2)
E   [right]: (1, 2)
----------------------------- Captured stdout call -----------------------------
[BACKEND] Using backend: numpy
╭───────────────────────── Understanding Your Results ─────────────────────────╮
│ Portfolio Analysis Results                                                   │
│ Metrics Explanation:                                                         │
│ • terminal_AnnReturn: annualised return (%)                                  │
│ • monthly_AnnVol: annualised volatility (%)                                  │
│ • monthly_VaR: Value at Risk (95% confidence)                                │
│ • monthly_BreachProb: Share of simulated months below the monthly breach     │
│ threshold                                                                    │
│ • terminal_ShortfallProb: Probability terminal compounded return is below    │
│ the annual threshold                                                         │
│ • monthly_MaxDD: Worst peak-to-trough decline of compounded wealth           │
│ • monthly_TimeUnderWater: Fraction of periods with compounded return below   │
│ zero                                                                         │
│ • monthly_TE: annualised active return volatility vs benchmark               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
┏━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┓
┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃
┡━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━━╇━━╇━━━╇━━╇━━━╇━━╇━━━╇━━╇━━━╇━━╇━━━╇━━╇━━━┩
│  │  │  │  │  │  │  │  │  │  │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │
│  │  │  │  │  │  │  │  │  │  │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │
│  │  │  │  │  │  │  │  │  │  │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┘

💡 Interpretation Tips:
• Lower terminal_ShortfallProb means fewer paths breach the terminal return 
threshold
• Higher terminal_AnnReturn with lower monthly_AnnVol indicates better 
risk-adjusted returns
• monthly_TE shows how volatile active returns are relative to the benchmark


🔍 Running sensitivity analysis...

📊 Sensitivity Analysis Results:
==================================================
📈 sigma_E_+5%          | Delta:  +0.0037%
📈 sigma_M_-5%          | Delta:  +0.0036%
📉 sigma_E_-5%          | Delta:  -0.0009%
📈 mu_H_+5%             | Delta:  +0.0008%
📉 mu_H_-5%             | Delta:  -0.0008%
📉 sigma_M_+5%          | Delta:  -0.0007%
📉 sigma_H_-5%          | Delta:  -0.0004%
📈 sigma_H_+5%          | Delta:  +0.0004%
📉 mu_E_+5%             | Delta:  +0.0000%
📉 mu_E_-5%             | Delta:  +0.0000%
📉 mu_M_+5%             | Delta:  +0.0000%
📉 mu_M_-5%             | Delta:  +0.0000%

✅ Sensitivity analysis completed. Evaluated 12 scenarios.
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_tradeoff_table[export]
1 failed, 17 warnings in 2.40s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_tradeoff_table[export]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.07s
exit=0
```

## test_single_run_tradeoff_table[failure]

Production mutation: `pa_core/cli.py`

```diff
-Trade-off table computation failed.
+Trade-off table warning suppressed.
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_tradeoff_table[failure]'
F                                                                        [100%]
=================================== FAILURES ===================================
___________________ test_single_run_tradeoff_table[failure] ____________________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:380: in test_single_run_tradeoff_table
    assert "Trade-off table computation failed" in capsys.readouterr().out
E   AssertionError: assert 'Trade-off table computation failed' in '[BACKEND] Using backend: numpy\n╭───────────────────────── Understanding Your Results ─────────────────────────╮\n│ P...a:  +0.0000%\n📉 mu_M_-5%             | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n'
E    +  where '[BACKEND] Using backend: numpy\n╭───────────────────────── Understanding Your Results ─────────────────────────╮\n│ P...a:  +0.0000%\n📉 mu_M_-5%             | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n╭───────────────────────── Understanding Your Results ─────────────...00%\n📉 mu_M_-5%             | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\n╭───────────────────────── Understanding Your Results ─────────────...00%\n📉 mu_M_-5%             | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x1666566c0>.readouterr
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_tradeoff_table[failure]
1 failed, 17 warnings in 2.06s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_tradeoff_table[failure]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.21s
exit=0
```

## test_single_run_registry_is_optional[registered]

Production mutation: `pa_core/cli.py`

```diff
-print(f"[REGISTRY] Scenario ID: {scenario_id}")
+print("registry ID lost")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_registry_is_optional[registered]'
F                                                                        [100%]
=================================== FAILURES ===================================
_______________ test_single_run_registry_is_optional[registered] _______________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:407: in test_single_run_registry_is_optional
    assert "[REGISTRY] Scenario ID: scenario-123" in capsys.readouterr().out
E   AssertionError: assert '[REGISTRY] Scenario ID: scenario-123' in '[BACKEND] Using backend: numpy\nregistry ID lost\n╭───────────────────────── Understanding Your Results ─────────────...a:  +0.0000%\n📉 mu_M_-5%             | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n'
E    +  where '[BACKEND] Using backend: numpy\nregistry ID lost\n╭───────────────────────── Understanding Your Results ─────────────...a:  +0.0000%\n📉 mu_M_-5%             | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n' = CaptureResult(out='[BACKEND] Using backend: numpy\nregistry ID lost\n╭───────────────────────── Understanding Your Res...00%\n📉 mu_M_-5%             | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n', err='').out
E    +    where CaptureResult(out='[BACKEND] Using backend: numpy\nregistry ID lost\n╭───────────────────────── Understanding Your Res...00%\n📉 mu_M_-5%             | Delta:  +0.0000%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n', err='') = readouterr()
E    +      where readouterr = <_pytest.capture.CaptureFixture object at 0x30c376690>.readouterr
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_registry_is_optional[registered]
1 failed, 17 warnings in 3.59s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_registry_is_optional[registered]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.31s
exit=0
```

## test_single_run_registry_is_optional[unavailable]

Production mutation: `pa_core/cli.py`

```diff
-logger.warning(f"Failed to register scenario: {exc}")
+logger.debug(f"Failed to register scenario: {exc}")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_registry_is_optional[unavailable]'
F                                                                        [100%]
=================================== FAILURES ===================================
______________ test_single_run_registry_is_optional[unavailable] _______________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:405: in test_single_run_registry_is_optional
    assert "Failed to register scenario: registry offline" in caplog.text
E   AssertionError: assert 'Failed to register scenario: registry offline' in ''
E    +  where '' = <_pytest.logging.LogCaptureFixture object at 0x3094539e0>.text
----------------------------- Captured stdout call -----------------------------
[BACKEND] Using backend: numpy
╭───────────────────────── Understanding Your Results ─────────────────────────╮
│ Portfolio Analysis Results                                                   │
│ Metrics Explanation:                                                         │
│ • terminal_AnnReturn: annualised return (%)                                  │
│ • monthly_AnnVol: annualised volatility (%)                                  │
│ • monthly_VaR: Value at Risk (95% confidence)                                │
│ • monthly_BreachProb: Share of simulated months below the monthly breach     │
│ threshold                                                                    │
│ • terminal_ShortfallProb: Probability terminal compounded return is below    │
│ the annual threshold                                                         │
│ • monthly_MaxDD: Worst peak-to-trough decline of compounded wealth           │
│ • monthly_TimeUnderWater: Fraction of periods with compounded return below   │
│ zero                                                                         │
│ • monthly_TE: annualised active return volatility vs benchmark               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
┏━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┳━━┳━━━┓
┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃  ┃ … ┃
┡━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━╇━━━╇━━╇━━━╇━━╇━━━╇━━╇━━━╇━━╇━━━╇━━╇━━━╇━━╇━━━┩
│  │  │  │  │  │  │  │  │  │  │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │
│  │  │  │  │  │  │  │  │  │  │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │
│  │  │  │  │  │  │  │  │  │  │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │  │ … │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┴──┴───┘

💡 Interpretation Tips:
• Lower terminal_ShortfallProb means fewer paths breach the terminal return 
threshold
• Higher terminal_AnnReturn with lower monthly_AnnVol indicates better 
risk-adjusted returns
• monthly_TE shows how volatile active returns are relative to the benchmark


🔍 Running sensitivity analysis...

📊 Sensitivity Analysis Results:
==================================================
📈 sigma_E_+5%          | Delta:  +0.0037%
📈 sigma_M_-5%          | Delta:  +0.0036%
📉 sigma_E_-5%          | Delta:  -0.0009%
📈 mu_H_+5%             | Delta:  +0.0008%
📉 mu_H_-5%             | Delta:  -0.0008%
📉 sigma_M_+5%          | Delta:  -0.0007%
📉 sigma_H_-5%          | Delta:  -0.0004%
📈 sigma_H_+5%          | Delta:  +0.0004%
📉 mu_E_+5%             | Delta:  +0.0000%
📉 mu_E_-5%             | Delta:  +0.0000%
📉 mu_M_+5%             | Delta:  +0.0000%
📉 mu_M_-5%             | Delta:  +0.0000%

✅ Sensitivity analysis completed. Evaluated 12 scenarios.
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_registry_is_optional[unavailable]
1 failed, 17 warnings in 2.19s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_registry_is_optional[unavailable]'  # restored
.                                                                        [100%]
1 passed, 17 warnings in 2.36s
exit=0
```

## test_single_run_suggestion_eof_aborts[grid-fallback]

Production mutation: `pa_core/cli.py`

```diff
-print(f"Optimizer unavailable; using grid fallback ({status}).")
+print("optimizer status lost")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_suggestion_eof_aborts[grid-fallback]'
F                                                                        [100%]
=================================== FAILURES ===================================
_____________ test_single_run_suggestion_eof_aborts[grid-fallback] _____________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:434: in test_single_run_suggestion_eof_aborts
    assert expected in text
E   AssertionError: assert 'Optimizer unavailable; using grid fallback' in '[BACKEND] Using backend: numpy\noptimizer status lost\n        optimizer_status\n0  grid_fallback:missing\nNo selection provided. Aborting run.\n'
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_suggestion_eof_aborts[grid-fallback]
1 failed, 4 warnings in 2.05s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_suggestion_eof_aborts[grid-fallback]'  # restored
.                                                                        [100%]
1 passed, 4 warnings in 1.84s
exit=0
```

## test_single_run_suggestion_eof_aborts[failed-fallback]

Production mutation: `pa_core/cli.py`

```diff
-print(f"Optimizer failed without grid fallback ({status}).")
+print("optimizer status lost")
```

Expected: the selected test reports FAILED (exit 1); restored: exit 0.

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_suggestion_eof_aborts[failed-fallback]'
F                                                                        [100%]
=================================== FAILURES ===================================
____________ test_single_run_suggestion_eof_aborts[failed-fallback] ____________
/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__closer-repair-3/tests/test_coverage_round_2375.py:434: in test_single_run_suggestion_eof_aborts
    assert expected in text
E   AssertionError: assert 'Optimizer failed without grid fallback' in '[BACKEND] Using backend: numpy\noptimizer status lost\n          optimizer_status\n0  fallback_failed:invalid\nNo selection provided. Aborting run.\n'
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_suggestion_eof_aborts[failed-fallback]
1 failed, 4 warnings in 2.06s
exit=1
```

```text
$ /opt/anaconda3/bin/python3 -m pytest -q --tb=short --disable-warnings 'tests/test_coverage_round_2375.py::test_single_run_suggestion_eof_aborts[failed-fallback]'  # restored
.                                                                        [100%]
1 passed, 4 warnings in 1.76s
exit=0
```
