# Mutation transcripts

Checked 2026-11-02 America/Chicago. Interpreter: `.venv/bin/python` (3.14.3).

## SKIP tests/test_config.py::test_sweep_parameter_rejects_invalid_definitions[kwargs0-requires values or min/max]

Pattern not found in `pa_core/config.py`.

## SKIP tests/test_config.py::test_sweep_config_rejects_invalid_sampling_combinations[kwargs0-parameters must not be empty]

Pattern not found in `pa_core/config.py`.

## test_return_unit_helpers_handle_non_mapping_and_regime_instances

Production: `pa_core/config.py`. Restored SHA-256: `e55c1fd119a655316ecadbd29c10fecb4365d36a40ce8421a0c09cfd7c6cf84c`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_config.py::test_return_unit_helpers_handle_non_mapping_and_regime_instances
.                                                                        [100%]
1 passed in 0.98s
exit=0
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_config.py::test_return_unit_helpers_handle_non_mapping_and_regime_instances
.                                                                        [100%]
1 passed in 1.12s
exit=0
```

## test_build_prompt_validates_inputs_and_includes_custom_rules

Production: `pa_core/llm/config_patch_chain.py`. Restored SHA-256: `42eb926c110931fe7f21b2df5e574aba5ef8522bfa57466856c931e01ad259c9`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_llm_config_patch_chain.py::test_build_prompt_validates_inputs_and_includes_custom_rules
>       with pytest.raises(TypeError, match="current_config"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE TypeError

tests/test_llm_config_patch_chain.py:21: Failed
=========================== short test summary info ============================
FAILED tests/test_llm_config_patch_chain.py::test_build_prompt_validates_inputs_and_includes_custom_rules
1 failed in 0.80s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_llm_config_patch_chain.py::test_build_prompt_validates_inputs_and_includes_custom_rules
.                                                                        [100%]
1 passed in 0.72s
exit=0
```

## test_parse_chain_output_accepts_json_yaml_and_normalizes_risk_flags[{"patch": {"set": {"n_simulations": 5}}}-[]]

Production: `pa_core/llm/config_patch_chain.py`. Restored SHA-256: `42eb926c110931fe7f21b2df5e574aba5ef8522bfa57466856c931e01ad259c9`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_llm_config_patch_chain.py::test_parse_chain_output_accepts_json_yaml_and_normalizes_risk_flags[{"patch": {"set": {"n_simulations": 5}}}-[]]

no tests ran in 0.23s
ERROR: not found: /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2382__opener/tests/test_llm_config_patch_chain.py::test_parse_chain_output_accepts_json_yaml_and_normalizes_risk_flags
(no match in any of [<Module test_llm_config_patch_chain.py>])

exit=4
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_llm_config_patch_chain.py::test_parse_chain_output_accepts_json_yaml_and_normalizes_risk_flags[{"patch": {"set": {"n_simulations": 5}}}-[]]
ERROR: not found: /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2382__opener/tests/test_llm_config_patch_chain.py::test_parse_chain_output_accepts_json_yaml_and_normalizes_risk_flags
(no match in any of [<Module test_llm_config_patch_chain.py>])

exit=4
```

## SKIP tests/test_llm_config_patch_chain.py::test_validation_error_includes_optional_type_metadata

Pattern not found in `pa_core/llm/config_patch_chain.py`.

## test_config_explicit_return_unit_and_regime_validation_branches

Production: `pa_core/config.py`. Restored SHA-256: `e55c1fd119a655316ecadbd29c10fecb4365d36a40ce8421a0c09cfd7c6cf84c`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_config_explicit_return_unit_and_regime_validation_branches
.                                                                        [100%]
1 passed in 0.70s
exit=0
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_config_explicit_return_unit_and_regime_validation_branches
.                                                                        [100%]
1 passed in 0.71s
exit=0
```

## test_loaders_cover_frequency_inference_and_csv_paths

Production: `pa_core/data/loaders.py`. Restored SHA-256: `628fc66d1eba50d8fcc4a1a8c2d699532fbe141825bc5e722078fbeb32aff6ff`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_loaders_cover_frequency_inference_and_csv_paths
E         
E         - unknown
E         + daily

tests/test_coverage_round_2382.py:122: AssertionError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2382.py::test_loaders_cover_frequency_inference_and_csv_paths
1 failed in 0.80s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_loaders_cover_frequency_inference_and_csv_paths
.                                                                        [100%]
1 passed in 0.78s
exit=0
```

## test_sweep_analysis_modes_and_runner_index_normalization

Production: `pa_core/sweep.py`. Restored SHA-256: `ff7c1189ea1ba53ffcea33eb27650a19f9002716c3e929e80532bb6a86d1060b`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_sweep_analysis_modes_and_runner_index_normalization
            )
>       raise ValueError(f"Unsupported analysis mode: {cfg.analysis_mode}")
E       ValueError: Unsupported analysis mode: vol_mult

pa_core/sweep.py:133: ValueError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2382.py::test_sweep_analysis_modes_and_runner_index_normalization
1 failed in 0.86s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_sweep_analysis_modes_and_runner_index_normalization
.                                                                        [100%]
1 passed in 0.76s
exit=0
```

## test_calibration_validation_and_constant_target

Production: `pa_core/calibration.py`. Restored SHA-256: `8cec3cb0fc79d069fc08246f74d3e5736b5876c85dcb1c06d08021bbf0e3ed7e`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_calibration_validation_and_constant_target
>       with pytest.raises(ValueError, match="returns data is empty"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_coverage_round_2382.py:280: Failed
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2382.py::test_calibration_validation_and_constant_target
1 failed in 0.84s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_calibration_validation_and_constant_target
.                                                                        [100%]
1 passed in 0.76s
exit=0
```

## test_config_regime_and_correlation_repair_edges

Production: `pa_core/config.py`. Restored SHA-256: `e55c1fd119a655316ecadbd29c10fecb4365d36a40ce8421a0c09cfd7c6cf84c`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_config_regime_and_correlation_repair_edges
>       with pytest.raises(ValueError, match="regimes must include at least one regime"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_coverage_round_2382.py:340: Failed
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2382.py::test_config_regime_and_correlation_repair_edges
1 failed in 0.83s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_config_regime_and_correlation_repair_edges
.                                                                        [100%]
1 passed in 0.76s
exit=0
```

## test_loaders_csv_variants_and_infer_freq_branches

Production: `pa_core/data/loaders.py`. Restored SHA-256: `628fc66d1eba50d8fcc4a1a8c2d699532fbe141825bc5e722078fbeb32aff6ff`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_loaders_csv_variants_and_infer_freq_branches
E         
E         Right contains 2 more items, first extra item: 'Return'
E         Use -v to get more diff

tests/test_coverage_round_2382.py:389: AssertionError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2382.py::test_loaders_csv_variants_and_infer_freq_branches
1 failed in 1.06s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_loaders_csv_variants_and_infer_freq_branches
.                                                                        [100%]
1 passed in 0.82s
exit=0
```

## test_sweep_run_paths_cover_regimes_cache_and_vol_mult

Production: `pa_core/sweep.py`. Restored SHA-256: `ff7c1189ea1ba53ffcea33eb27650a19f9002716c3e929e80532bb6a86d1060b`.

RED

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_sweep_run_paths_cover_regimes_cache_and_vol_mult
>           sigma_h = float(regime.sigma_H) if regime.sigma_H is not None else float(cfg.sigma_H)
                                               ^^^^^^^^^^^^^^
E           AttributeError: 'dict' object has no attribute 'sigma_H'

pa_core/sim/regimes.py:36: AttributeError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2382.py::test_sweep_run_paths_cover_regimes_cache_and_vol_mult
1 failed in 0.88s
exit=1
```

GREEN

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2382.py::test_sweep_run_paths_cover_regimes_cache_and_vol_mult
.                                                                        [100%]
1 passed in 0.88s
exit=0
```
