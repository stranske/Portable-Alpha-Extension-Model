# Mutation transcripts

Checked 2026-11-02 America/Chicago on base 693037352463bb4d4db215144aaa9f2d2eb3a224.
Interpreter: repository .venv/bin/python (3.14.3), current requirements.lock.
Concise excerpts: command, named failing assertion/error, terminal result, and exit code. Warnings and unrelated traceback context omitted.

## 1. test_result_helpers_preserve_json_lists_and_prefer_total_metric_row

Production: `pa_core/llm/result_explain.py`. Restored SHA-256: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841`.

Mutation:

```diff
-if isinstance(value, list):
-        return [_to_json_safe(v) for v in value]
+if isinstance(value, list):
+        return []
```

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_result_helpers_preserve_json_lists_and_prefer_total_metric_row
>       assert result_explain._to_json_safe(value) == ["2026-01-01T00:00:00", None]
E       AssertionError: assert [] == ['2026-01-01T00:00:00', None]
E         
E         Right contains 2 more items, first extra item: '2026-01-01T00:00:00'
E         Use -v to get more diff
tests/test_coverage_round_2421.py:40: AssertionError
FAILED tests/test_coverage_round_2421.py::test_result_helpers_preserve_json_lists_and_prefer_total_metric_row
1 failed, 14 warnings in 1.32s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_result_helpers_preserve_json_lists_and_prefer_total_metric_row
1 passed, 14 warnings in 0.89s
exit=0
```

## 2. test_result_explain_tolerates_llm_path_telemetry_failure_and_non_mapping_summary

Production: `pa_core/llm/result_explain.py`. Restored SHA-256: `7aa78795d0cbdcf65de0a218f0f890f1f0aac3398c732d452e159dc25bca9841`.

Mutation:

```diff
-except Exception:
-            logger.warning("Failed to record result-explanation fleet event", exc_info=True)
-    return
+except Exception:
+            payload["status"] = "error"
+    return
```

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_result_explain_tolerates_llm_path_telemetry_failure_and_non_mapping_summary
>       assert payload["status"] == "success"
E       AssertionError: assert 'error' == 'success'
E         
E         - success
E         + error
tests/test_coverage_round_2421.py:75: AssertionError
FAILED tests/test_coverage_round_2421.py::test_result_explain_tolerates_llm_path_telemetry_failure_and_non_mapping_summary
1 failed, 14 warnings in 1.18s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_result_explain_tolerates_llm_path_telemetry_failure_and_non_mapping_summary
1 passed, 14 warnings in 1.00s
exit=0
```

## 3. test_units_cover_monthly_to_annual_covariance_and_identity_series

Production: `pa_core/units.py`. Restored SHA-256: `4dac096784e2f79f663e0daf42a7ad037ee19a4bae55a5c43d02d573ff1810b5`.

Mutation:

```diff
-return np.asarray(cov, dtype=float) * MONTHS_PER_YEAR
+return np.asarray(cov, dtype=float)
```

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_units_cover_monthly_to_annual_covariance_and_identity_series
>       assert np.allclose(
E       AssertionError: assert False
E        +  where False = <function allclose at 0x1096f5eb0>(array([[0.08333333, 0.        ],\n       [0.        , 0.08333333]]), array([[1., 0.],\n       [0., 1.]]))
E        +    where <function allclose at 0x1096f5eb0> = np.allclose
E        +    and   array([[0.08333333, 0.        ],\n       [0.        , 0.08333333]]) = <function convert_covariance at 0x10ecfbed0>(array([[0.08333333, 0.        ],\n       [0.        , 0.08333333]]), from_unit='monthly', to_unit='annual')
E        +      where <function convert_covariance at 0x10ecfbed0> = units.convert_covariance
E        +    and   array([[1., 0.],\n       [0., 1.]]) = <function eye at 0x1097d8720>(2)
E        +      where <function eye at 0x1097d8720> = np.eye
tests/test_coverage_round_2421.py:84: AssertionError
FAILED tests/test_coverage_round_2421.py::test_units_cover_monthly_to_annual_covariance_and_identity_series
1 failed, 14 warnings in 1.07s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_units_cover_monthly_to_annual_covariance_and_identity_series
1 passed, 14 warnings in 0.89s
exit=0
```

## 4. test_internal_financing_uses_default_rng_when_stochastic_rng_is_omitted

Production: `pa_core/sim/internal_pa_financing.py`. Restored SHA-256: `40d4cad40f4ae20195787d9e0ea64ecf057c8a640dfbd2bfb0e02e3fa8a667ba`.

Mutation:

```diff
-rng = spawn_rngs(None, 1)[0]
+rng = spawn_rngs(123, 1)[0]
```

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_internal_financing_uses_default_rng_when_stochastic_rng_is_omitted
>       assert calls == [(None, 1)]
E       assert [(123, 1)] == [(None, 1)]
E         
E         At index 0 diff: (123, 1) != (None, 1)
E         Use -v to get more diff
tests/test_coverage_round_2421.py:119: AssertionError
FAILED tests/test_coverage_round_2421.py::test_internal_financing_uses_default_rng_when_stochastic_rng_is_omitted
1 failed, 14 warnings in 0.99s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_internal_financing_uses_default_rng_when_stochastic_rng_is_omitted
1 passed, 14 warnings in 1.13s
exit=0
```

## 5. test_metrics_reject_invalid_inputs_and_handle_small_samples

Production: `pa_core/sim/metrics.py`. Restored SHA-256: `8c828e8525a61c2606824b825e90711fb642c334670661fa8e652fa16ae58599`.

Mutation:

```diff
-if not 0 < confidence < 1:
+if not 0 < confidence <= 1:
```

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_metrics_reject_invalid_inputs_and_handle_small_samples
>       with pytest.raises(ValueError, match="confidence"):
E       Failed: DID NOT RAISE ValueError
tests/test_coverage_round_2421.py:125: Failed
FAILED tests/test_coverage_round_2421.py::test_metrics_reject_invalid_inputs_and_handle_small_samples
1 failed, 14 warnings in 1.19s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_metrics_reject_invalid_inputs_and_handle_small_samples
1 passed, 14 warnings in 0.98s
exit=0
```

## 6. test_convergence_diagnostic_rejects_invalid_requests

Production: `pa_core/sim/metrics.py`. Restored SHA-256: `8c828e8525a61c2606824b825e90711fb642c334670661fa8e652fa16ae58599`.

Mutation:

```diff
-checkpoint <= 0.0 or checkpoint > 1.0
+checkpoint < 0.0 or checkpoint > 1.0
```

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_convergence_diagnostic_rejects_invalid_requests
>       with pytest.raises(ValueError, match="interval"):
E       Failed: DID NOT RAISE ValueError
tests/test_coverage_round_2421.py:158: Failed
FAILED tests/test_coverage_round_2421.py::test_convergence_diagnostic_rejects_invalid_requests
1 failed, 14 warnings in 1.21s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_convergence_diagnostic_rejects_invalid_requests
1 passed, 14 warnings in 1.00s
exit=0
```

## 7. test_metrics_cover_registration_plugins_and_boundary_paths

Production: `pa_core/sim/metrics.py`. Restored SHA-256: `8c828e8525a61c2606824b825e90711fb642c334670661fa8e652fa16ae58599`.

Mutation:

```diff
-register_metric(ep.name, ep.load())
+ep.load()
```

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_metrics_cover_registration_plugins_and_boundary_paths
>       assert plugin_name in _EXTRA_METRICS
E       AssertionError: assert 'CoverageRound2421_PluginMetric' in {'CoverageRound2421_DuplicateMetric': <function test_metrics_cover_registration_plugins_and_boundary_paths.<locals>._metric at 0x10ed105c0>}
tests/test_coverage_round_2421.py:201: AssertionError
FAILED tests/test_coverage_round_2421.py::test_metrics_cover_registration_plugins_and_boundary_paths
1 failed, 14 warnings in 1.17s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2421.py::test_metrics_cover_registration_plugins_and_boundary_paths
1 passed, 14 warnings in 1.00s
exit=0
```
