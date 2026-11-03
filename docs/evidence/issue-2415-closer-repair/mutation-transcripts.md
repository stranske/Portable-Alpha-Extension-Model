# Missing deliberate-break transcripts

Check date: 2026-11-03 (America/Chicago). Full output; one mutation at a time.

## regime-timeline

File: `pa_core/viz/compare_scenarios.py`.

Before:
```python
        output["regime_timeline"] = _make_regime_timeline(regime_rows)
```

Temporary mutation:
```python
        pass  # deliberate break: suppress regime_timeline
```

RED:
```text
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_compare_scenarios_coverage_2415.py::test_compare_scenarios_builds_all_optional_figures_from_mapping_inputs
F                                                                        [100%]
=================================== FAILURES ===================================
____ test_compare_scenarios_builds_all_optional_figures_from_mapping_inputs ____

    def test_compare_scenarios_builds_all_optional_figures_from_mapping_inputs() -> None:
        scenarios = [
            {
                "combination_id": 7,
                "summary": _summary(),
                "raw_returns": {
                    "Base": pd.DataFrame([[0.01, -0.02], [0.03, 0.01]]),
                    "Regime": [["normal", "stress"], ["normal", "normal"]],
                },
            },
            {
                "label": "Alternative",
                "summary": _summary("Total"),
                "returns": [0.01, 0.02, -0.01],
                "regime_labels": {"Regime": ["stress", "normal", "normal"]},
            },
        ]

        figures = module.compare_scenarios(scenarios)

>       assert set(figures) == {
            "risk_return",
            "cvar_return",
            "return_distribution",
            "risk_metrics",
            "regime_timeline",
        }
E       AssertionError: assert {'cvar_return...'risk_return'} == {'cvar_return...'risk_return'}
E
E         Extra items in the right set:
E         'regime_timeline'
E         Use -v to get more diff

tests/test_compare_scenarios_coverage_2415.py:47: AssertionError
=========================== short test summary info ============================
FAILED tests/test_compare_scenarios_coverage_2415.py::test_compare_scenarios_builds_all_optional_figures_from_mapping_inputs
1 failed in 1.02s

exit=1
```

Restored PASS (byte-identical SHA-256 `0e2b0bff6ce1dd207659953f0707f6b8c4c1a314846e1eecf8dbf517d7a44743`):
```text
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_compare_scenarios_coverage_2415.py::test_compare_scenarios_builds_all_optional_figures_from_mapping_inputs
.                                                                        [100%]
1 passed in 0.85s

exit=0
```

## risk-metrics

File: `pa_core/viz/compare_scenarios.py`.

Before:
```python
    output["risk_metrics"] = _make_risk_metric_bars(compare_df)
```

Temporary mutation:
```python
    pass  # deliberate break: suppress risk_metrics
```

RED:
```text
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_compare_scenarios_coverage_2415.py::test_compare_scenarios_supports_summary_frames_and_reports_invalid_inputs
F                                                                        [100%]
=================================== FAILURES ===================================
__ test_compare_scenarios_supports_summary_frames_and_reports_invalid_inputs ___

    def test_compare_scenarios_supports_summary_frames_and_reports_invalid_inputs() -> None:
        figures = module.compare_scenarios([_summary()], include_returns=False)
>       assert set(figures) == {"risk_return", "cvar_return", "risk_metrics"}
E       AssertionError: assert {'cvar_return', 'risk_return'} == {'cvar_return...'risk_return'}
E
E         Extra items in the right set:
E         'risk_metrics'
E         Use -v to get more diff

tests/test_compare_scenarios_coverage_2415.py:61: AssertionError
=========================== short test summary info ============================
FAILED tests/test_compare_scenarios_coverage_2415.py::test_compare_scenarios_supports_summary_frames_and_reports_invalid_inputs
1 failed in 0.92s

exit=1
```

Restored PASS (byte-identical SHA-256 `0e2b0bff6ce1dd207659953f0707f6b8c4c1a314846e1eecf8dbf517d7a44743`):
```text
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_compare_scenarios_coverage_2415.py::test_compare_scenarios_supports_summary_frames_and_reports_invalid_inputs
.                                                                        [100%]
1 passed in 0.95s

exit=0
```

## agents-fallback

File: `pa_core/reporting/excel.py`.

Before:
```python
    df = build_agent_semantics(cast(ModelConfig, cfg))
    inputs_dict["_agent_semantics_df"] = _records_to_builtin(df.to_dict(orient="records"))
    return df
```

Temporary mutation:
```python
    df = build_agent_semantics(cast(ModelConfig, cfg))
    inputs_dict["_agent_semantics_df"] = _records_to_builtin(df.to_dict(orient="records"))
    return None  # deliberate break: discard agents-derived fallback
```

RED:
```text
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2415.py::test_excel_builds_agent_semantics_from_agents_when_frame_missing
F                                                                        [100%]
=================================== FAILURES ===================================
_______ test_excel_builds_agent_semantics_from_agents_when_frame_missing _______

    def test_excel_builds_agent_semantics_from_agents_when_frame_missing() -> None:
        agents = [
            {
                "name": "Base",
                "capital": 100.0,
                "beta_share": 1.0,
                "alpha_share": 0.0,
                "extra": {},
            }
        ]
        inputs = {
            "agents": agents,
            "total_fund_capital": 100.0,
            "reference_sigma": 0.15,
            "volatility_multiple": 1.0,
            "financing_model": "flat",
        }
        df = _ensure_agent_semantics_df(inputs)
>       assert df is not None
E       assert None is not None

tests/test_coverage_round_2415.py:122: AssertionError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2415.py::test_excel_builds_agent_semantics_from_agents_when_frame_missing
1 failed in 0.97s

exit=1
```

Restored PASS (byte-identical SHA-256 `9aac72804481848252673e177ab0a85b0d5240f178084021d1b7992f569a2f41`):
```text
$ /Users/teacher/Library/CloudStorage/Dropbox/Learning/Code/Portable-Alpha-Extension-Model/.venv/bin/python -m pytest -q tests/test_coverage_round_2415.py::test_excel_builds_agent_semantics_from_agents_when_frame_missing
.                                                                        [100%]
1 passed in 0.85s

exit=0
```
