# Mutation transcripts

Check date: 2026-11-02 America/Chicago.

## test_compare_run_helpers_reject_unusable_artifacts

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts
F                                                                        [100%]
=================================== FAILURES ===================================
______________ test_compare_run_helpers_reject_unusable_artifacts ______________

tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/cursor/tmp/pytest-of-teacher/pytest-557/test_compare_run_helpers_rejec0')

    def test_compare_run_helpers_reject_unusable_artifacts(tmp_path) -> None:
        """Unreadable or structurally invalid artifacts must not look usable."""
        non_mapping_path = tmp_path / "not-a-manifest.json"
        non_mapping_path.write_text(json.dumps(["not", "a", "mapping"]))
    
        assert load_prior_manifest(None) == (None, None)
>       assert load_prior_manifest({"previous_run": str(non_mapping_path)}) == (
            None,
            non_mapping_path,
        )
E       AssertionError: assert ({}, PosixPat...nifest.json')) == (None, PosixP...nifest.json'))
E         
E         At index 0 diff: {} != None
E         Use -v to get more diff

tests/test_coverage_round_2375.py:33: AssertionError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts
1 failed in 0.74s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_compare_run_helpers_reject_unusable_artifacts  # restored
.                                                                        [100%]
1 passed in 0.70s
exit=0
```

## test_compare_runs_explains_missing_prior_summary

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_compare_runs_explains_missing_prior_summary
F                                                                        [100%]
=================================== FAILURES ===================================
_______________ test_compare_runs_explains_missing_prior_summary _______________

tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/cursor/tmp/pytest-of-teacher/pytest-559/test_compare_runs_explains_mis0')

    def test_compare_runs_explains_missing_prior_summary(tmp_path) -> None:
        """A valid prior manifest without a workbook is a distinct user error."""
        prior_manifest_path = tmp_path / "prior-manifest.json"
        prior_manifest_path.write_text(json.dumps({"cli_args": {"output": "missing.xlsx"}}))
    
        with pytest.raises(ValueError, match="No readable prior summary found"):
>           compare_runs(
                current_summary=pd.DataFrame({"monthly_TE": [0.01]}),
                current_manifest={"previous_run": str(prior_manifest_path)},
                questions="What changed?",
            )

tests/test_coverage_round_2375.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
pa_core/llm/compare_runs.py:406: in compare_runs
    metrics_b = build_metric_catalog(prior_summary)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

summary_df = None

    def build_metric_catalog(summary_df: pd.DataFrame) -> dict[str, float]:
        """Extract compact metrics used by compare/explain output."""
    
        if not isinstance(summary_df, pd.DataFrame):
>           raise TypeError("summary_df must be a pandas DataFrame")
E           TypeError: summary_df must be a pandas DataFrame

pa_core/llm/compare_runs.py:295: TypeError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_compare_runs_explains_missing_prior_summary
1 failed in 0.79s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_compare_runs_explains_missing_prior_summary  # restored
.                                                                        [100%]
1 passed in 0.66s
exit=0
```

## test_compare_runs_rejects_missing_prior_manifest

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_compare_runs_rejects_missing_prior_manifest
F                                                                        [100%]
=================================== FAILURES ===================================
_______________ test_compare_runs_rejects_missing_prior_manifest _______________

    def test_compare_runs_rejects_missing_prior_manifest() -> None:
        """A request without a previous manifest cannot produce a comparison."""
>       with pytest.raises(ValueError, match="No readable prior manifest"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AssertionError: Regex pattern did not match.
E         Expected regex: 'No readable prior manifest'
E         Actual message: 'No readable prior summary found at expected path: <unset>'

tests/test_coverage_round_2375.py:56: AssertionError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_compare_runs_rejects_missing_prior_manifest
1 failed in 0.72s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_compare_runs_rejects_missing_prior_manifest  # restored
.                                                                        [100%]
1 passed in 0.67s
exit=0
```

## test_compare_runs_survives_fleet_event_failure

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_compare_runs_survives_fleet_event_failure
F                                                                        [100%]
=================================== FAILURES ===================================
________________ test_compare_runs_survives_fleet_event_failure ________________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11a67c770>

    def test_compare_runs_survives_fleet_event_failure(monkeypatch: pytest.MonkeyPatch) -> None:
        """Telemetry failure must not discard an otherwise valid comparison."""
        prior_summary = pd.DataFrame({"monthly_TE": [0.01]})
        monkeypatch.setattr(
            "pa_core.llm.compare_runs.load_prior_manifest",
            lambda _manifest: ({"seed": 1, "cli_args": {"output": "prior.xlsx"}}, Path("prior.json")),
        )
        monkeypatch.setattr(
            "pa_core.llm.compare_runs.load_prior_summary",
            lambda _manifest, *, manifest_path: (prior_summary, Path("prior.xlsx")),
        )
        monkeypatch.setattr(
            "pa_core.llm.compare_runs._invoke_comparison_llm",
            lambda *_args, **_kwargs: "comparison complete",
        )
        monkeypatch.setattr(
            "pa_core.llm.compare_runs.record_fleet_event",
            lambda _event: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
        )
    
>       text, _, payload = compare_runs(
            current_summary=pd.DataFrame({"monthly_TE": [0.02]}),
            current_manifest={"seed": 2},
            questions="What changed?",
        )

tests/test_coverage_round_2375.py:84: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
pa_core/llm/compare_runs.py:476: in compare_runs
    record_fleet_event(
tests/test_coverage_round_2375.py:81: in <lambda>
    lambda _event: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

.0 = <tuple_iterator object at 0x102de8460>

>       lambda _event: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
    )
E   RuntimeError: telemetry unavailable

tests/test_coverage_round_2375.py:81: RuntimeError
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_compare_runs_survives_fleet_event_failure
1 failed in 0.72s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_compare_runs_survives_fleet_event_failure  # restored
.                                                                        [100%]
1 passed in 0.72s
exit=0
```

## test_metric_catalog_rejects_non_dataframe_and_empty_delta

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta
F                                                                        [100%]
=================================== FAILURES ===================================
__________ test_metric_catalog_rejects_non_dataframe_and_empty_delta ___________

    def test_metric_catalog_rejects_non_dataframe_and_empty_delta() -> None:
        """Comparison input validation cannot silently turn invalid inputs into metrics."""
>       with pytest.raises(TypeError, match="pandas DataFrame"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE TypeError

tests/test_coverage_round_2375.py:96: Failed
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta
1 failed in 0.78s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_metric_catalog_rejects_non_dataframe_and_empty_delta  # restored
.                                                                        [100%]
1 passed in 0.66s
exit=0
```

## test_single_run_packet_failure (RuntimeError)

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable] _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x1184589d0>
tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/cursor/tmp/pytest-of-teacher/pytest-561/test_single_run_packet_failure0')
capsys = <_pytest.capture.CaptureFixture object at 0x1076274d0>
error_type = <class 'RuntimeError'>
expected = 'Export packet failed: packet unavailable'

    @pytest.mark.parametrize(
        ("error_type", "expected"),
        [
            (RuntimeError, "Export packet failed: packet unavailable"),
            (ImportError, "Export packet failed due to missing dependency: packet unavailable"),
            (ValueError, "Export packet failed due to data or configuration issue: packet unavailable"),
            (OSError, "Export packet failed due to file system issue: packet unavailable"),
        ],
    )
    def test_single_run_packet_failure_reports_the_failure_class(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        error_type: type[Exception],
        expected: str,
    ) -> None:
        """Packet failures must be specific and must finish the run cleanly."""
        monkeypatch.chdir(tmp_path)
        _install_fake_viz(monkeypatch)
        config, index, output = _write_single_run_config(tmp_path)
    
        def fail_packet(**_kwargs: object) -> tuple[Path, Path]:
            raise error_type("packet unavailable")
    
        monkeypatch.setattr("pa_core.reporting.export_packet.create_export_packet", fail_packet)
    
        main(_single_run_argv(config, index, output, "--packet"))
    
>       assert f"❌ {expected}" in capsys.readouterr().out
E       AssertionError: assert '❌ Export packet failed: packet unavailable' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n❌ Export packet failed (mutated): packet unavailable\n'
E        +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...%\n\n✅ Sensitivity analysis completed. Evaluated 12 scenarios.\n❌ Export packet failed (mutated): packet unavailable\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...nsitivity analysis completed. Evaluated 12 scenarios.\n❌ Export packet failed (mutated): packet unavailable\n', err='').out
E        +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...nsitivity analysis completed. Evaluated 12 scenarios.\n❌ Export packet failed (mutated): packet unavailable\n', err='') = readouterr()
E        +      where readouterr = <_pytest.capture.CaptureFixture object at 0x1076274d0>.readouterr

/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:183: AssertionError
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:181: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--packet"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]
1 failed, 15 warnings in 1.07s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]  # restored
.                                                                        [100%]
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[RuntimeError-Export packet failed: packet unavailable]
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:181: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--packet"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 15 warnings in 1.01s
exit=0
```

## test_single_run_packet_failure (ImportError)

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable] _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11be949d0>
tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/cursor/tmp/pytest-of-teacher/pytest-563/test_single_run_packet_failure0')
capsys = <_pytest.capture.CaptureFixture object at 0x10b7874d0>
error_type = <class 'ImportError'>
expected = 'Export packet failed due to missing dependency: packet unavailable'

    @pytest.mark.parametrize(
        ("error_type", "expected"),
        [
            (RuntimeError, "Export packet failed: packet unavailable"),
            (ImportError, "Export packet failed due to missing dependency: packet unavailable"),
            (ValueError, "Export packet failed due to data or configuration issue: packet unavailable"),
            (OSError, "Export packet failed due to file system issue: packet unavailable"),
        ],
    )
    def test_single_run_packet_failure_reports_the_failure_class(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        error_type: type[Exception],
        expected: str,
    ) -> None:
        """Packet failures must be specific and must finish the run cleanly."""
        monkeypatch.chdir(tmp_path)
        _install_fake_viz(monkeypatch)
        config, index, output = _write_single_run_config(tmp_path)
    
        def fail_packet(**_kwargs: object) -> tuple[Path, Path]:
            raise error_type("packet unavailable")
    
        monkeypatch.setattr("pa_core.reporting.export_packet.create_export_packet", fail_packet)
    
        main(_single_run_argv(config, index, output, "--packet"))
    
>       assert f"❌ {expected}" in capsys.readouterr().out
E       AssertionError: assert '❌ Export packet failed due to missing dependency: packet unavailable' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...pendency (mutated): packet unavailable\n💡 Install required packages: pip install plotly kaleido openpyxl python-pptx\n'
E        +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...pendency (mutated): packet unavailable\n💡 Install required packages: pip install plotly kaleido openpyxl python-pptx\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...(mutated): packet unavailable\n💡 Install required packages: pip install plotly kaleido openpyxl python-pptx\n', err='').out
E        +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...(mutated): packet unavailable\n💡 Install required packages: pip install plotly kaleido openpyxl python-pptx\n', err='') = readouterr()
E        +      where readouterr = <_pytest.capture.CaptureFixture object at 0x10b7874d0>.readouterr

/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:183: AssertionError
------------------------------ Captured log call -------------------------------
ERROR    pa_core.cli:cli.py:2024 Export packet failed due to missing dependency: packet unavailable
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:181: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--packet"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]
1 failed, 15 warnings in 0.95s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]  # restored
.                                                                        [100%]
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ImportError-Export packet failed due to missing dependency: packet unavailable]
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:181: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--packet"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 15 warnings in 1.01s
exit=0
```

## test_single_run_packet_failure (ValueError)

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable] _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x1199149d0>
tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/cursor/tmp/pytest-of-teacher/pytest-565/test_single_run_packet_failure0')
capsys = <_pytest.capture.CaptureFixture object at 0x108b6f4d0>
error_type = <class 'ValueError'>
expected = 'Export packet failed due to data or configuration issue: packet unavailable'

    @pytest.mark.parametrize(
        ("error_type", "expected"),
        [
            (RuntimeError, "Export packet failed: packet unavailable"),
            (ImportError, "Export packet failed due to missing dependency: packet unavailable"),
            (ValueError, "Export packet failed due to data or configuration issue: packet unavailable"),
            (OSError, "Export packet failed due to file system issue: packet unavailable"),
        ],
    )
    def test_single_run_packet_failure_reports_the_failure_class(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        error_type: type[Exception],
        expected: str,
    ) -> None:
        """Packet failures must be specific and must finish the run cleanly."""
        monkeypatch.chdir(tmp_path)
        _install_fake_viz(monkeypatch)
        config, index, output = _write_single_run_config(tmp_path)
    
        def fail_packet(**_kwargs: object) -> tuple[Path, Path]:
            raise error_type("packet unavailable")
    
        monkeypatch.setattr("pa_core.reporting.export_packet.create_export_packet", fail_packet)
    
        main(_single_run_argv(config, index, output, "--packet"))
    
>       assert f"❌ {expected}" in capsys.readouterr().out
E       AssertionError: assert '❌ Export packet failed due to data or configuration issue: packet unavailable' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ue to data or configuration issue (mutated): packet unavailable\n💡 Check your data inputs and configuration settings\n'
E        +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ue to data or configuration issue (mutated): packet unavailable\n💡 Check your data inputs and configuration settings\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...a or configuration issue (mutated): packet unavailable\n💡 Check your data inputs and configuration settings\n', err='').out
E        +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...a or configuration issue (mutated): packet unavailable\n💡 Check your data inputs and configuration settings\n', err='') = readouterr()
E        +      where readouterr = <_pytest.capture.CaptureFixture object at 0x108b6f4d0>.readouterr

/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:183: AssertionError
------------------------------ Captured log call -------------------------------
ERROR    pa_core.cli:cli.py:2032 Export packet failed due to data/config issue: packet unavailable
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:181: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--packet"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]
1 failed, 15 warnings in 0.97s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]  # restored
.                                                                        [100%]
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[ValueError-Export packet failed due to data or configuration issue: packet unavailable]
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:181: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--packet"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 15 warnings in 0.93s
exit=0
```

## test_single_run_packet_failure (OSError)

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable] _

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11a8e49d0>
tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/cursor/tmp/pytest-of-teacher/pytest-567/test_single_run_packet_failure0')
capsys = <_pytest.capture.CaptureFixture object at 0x109a5b4d0>
error_type = <class 'OSError'>
expected = 'Export packet failed due to file system issue: packet unavailable'

    @pytest.mark.parametrize(
        ("error_type", "expected"),
        [
            (RuntimeError, "Export packet failed: packet unavailable"),
            (ImportError, "Export packet failed due to missing dependency: packet unavailable"),
            (ValueError, "Export packet failed due to data or configuration issue: packet unavailable"),
            (OSError, "Export packet failed due to file system issue: packet unavailable"),
        ],
    )
    def test_single_run_packet_failure_reports_the_failure_class(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        error_type: type[Exception],
        expected: str,
    ) -> None:
        """Packet failures must be specific and must finish the run cleanly."""
        monkeypatch.chdir(tmp_path)
        _install_fake_viz(monkeypatch)
        config, index, output = _write_single_run_config(tmp_path)
    
        def fail_packet(**_kwargs: object) -> tuple[Path, Path]:
            raise error_type("packet unavailable")
    
        monkeypatch.setattr("pa_core.reporting.export_packet.create_export_packet", fail_packet)
    
        main(_single_run_argv(config, index, output, "--packet"))
    
>       assert f"❌ {expected}" in capsys.readouterr().out
E       AssertionError: assert '❌ Export packet failed due to file system issue: packet unavailable' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ket failed due to file system issue (mutated): packet unavailable\n💡 Check file permissions and available disk space\n'
E        +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============...ket failed due to file system issue (mutated): packet unavailable\n💡 Check file permissions and available disk space\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...d due to file system issue (mutated): packet unavailable\n💡 Check file permissions and available disk space\n', err='').out
E        +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...d due to file system issue (mutated): packet unavailable\n💡 Check file permissions and available disk space\n', err='') = readouterr()
E        +      where readouterr = <_pytest.capture.CaptureFixture object at 0x109a5b4d0>.readouterr

/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:183: AssertionError
------------------------------ Captured log call -------------------------------
ERROR    pa_core.cli:cli.py:2038 Export packet failed due to file system issue: packet unavailable
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:181: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--packet"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]
1 failed, 15 warnings in 1.07s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]  # restored
.                                                                        [100%]
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_packet_failure_reports_the_failure_class[OSError-Export packet failed due to file system issue: packet unavailable]
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:181: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--packet"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 15 warnings in 0.95s
exit=0
```

## test_single_run_gif_conversion_error_is_reported

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported
F                                                                        [100%]
=================================== FAILURES ===================================
_______________ test_single_run_gif_conversion_error_is_reported _______________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11a2e49d0>
tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/cursor/tmp/pytest-of-teacher/pytest-569/test_single_run_gif_conversion0')
capsys = <_pytest.capture.CaptureFixture object at 0x1093bb230>

    def test_single_run_gif_conversion_error_is_reported(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Invalid return arrays must not proceed to an animation export."""
        monkeypatch.chdir(tmp_path)
        _install_fake_viz(monkeypatch)
        config, index, output = _write_single_run_config(tmp_path)
        # `pa_core.viz` is a fake module here, so the real utils module is patched by object, not by path.
        monkeypatch.setattr(
            viz_utils,
            "safe_to_numpy",
            lambda _value: (_ for _ in ()).throw(ValueError("not numeric")),
        )
    
        main(_single_run_argv(config, index, output, "--gif"))
    
>       assert "❌ GIF export failed: Data conversion error - not numeric" in capsys.readouterr().out
E       AssertionError: assert '❌ GIF export failed: Data conversion error - not numeric' in '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============... export failed: Data conversion error (mutated) - not numeric\n💡 Check that return data contains only numeric values\n'
E        +  where '[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============... export failed: Data conversion error (mutated) - not numeric\n💡 Check that return data contains only numeric values\n' = CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...ailed: Data conversion error (mutated) - not numeric\n💡 Check that return data contains only numeric values\n', err='').out
E        +    where CaptureResult(out='[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...ailed: Data conversion error (mutated) - not numeric\n💡 Check that return data contains only numeric values\n', err='') = readouterr()
E        +      where readouterr = <_pytest.capture.CaptureFixture object at 0x1093bb230>.readouterr

/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:202: AssertionError
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:200: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--gif"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported
1 failed, 15 warnings in 0.99s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported  # restored
.                                                                        [100%]
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_gif_conversion_error_is_reported
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:200: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--gif"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 15 warnings in 0.92s
exit=0
```

## test_single_run_dashboard_missing_file_is_reported

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported
F                                                                        [100%]
=================================== FAILURES ===================================
______________ test_single_run_dashboard_missing_file_is_reported ______________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x1176089d0>
tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/cursor/tmp/pytest-of-teacher/pytest-571/test_single_run_dashboard_miss0')
capsys = <_pytest.capture.CaptureFixture object at 0x106f8f230>

    def test_single_run_dashboard_missing_file_is_reported(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The dashboard path is checked before any launcher is invoked."""
        monkeypatch.chdir(tmp_path)
        _install_fake_viz(monkeypatch)
        config, index, output = _write_single_run_config(tmp_path)
    
        main(_single_run_argv(config, index, output, "--dashboard"))
    
>       assert "❌ Dashboard launch failed: Dashboard file not found" in capsys.readouterr().out
E       assert '❌ Dashboard launch failed: Dashboard file not found' in "[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============... Dashboard file not found: dashboard/app.py\n💡 Ensure the dashboard files are present in the 'dashboard/' directory.\n"
E        +  where "[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Results:\n=============... Dashboard file not found: dashboard/app.py\n💡 Ensure the dashboard files are present in the 'dashboard/' directory.\n" = CaptureResult(out="[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...d file not found: dashboard/app.py\n💡 Ensure the dashboard files are present in the 'dashboard/' directory.\n", err='').out
E        +    where CaptureResult(out="[BACKEND] Using backend: numpy\n\n🔍 Running sensitivity analysis...\n\n📊 Sensitivity Analysis Resul...d file not found: dashboard/app.py\n💡 Ensure the dashboard files are present in the 'dashboard/' directory.\n", err='') = readouterr()
E        +      where readouterr = <_pytest.capture.CaptureFixture object at 0x106f8f230>.readouterr

/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:215: AssertionError
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:213: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--dashboard"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported
1 failed, 15 warnings in 1.14s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported  # restored
.                                                                        [100%]
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_dashboard_missing_file_is_reported
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:213: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--dashboard"))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 15 warnings in 0.90s
exit=0
```

## test_single_run_bundle_contains_the_completed_artifacts

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts
F                                                                        [100%]
=================================== FAILURES ===================================
___________ test_single_run_bundle_contains_the_completed_artifacts ____________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x11a854770>
tmp_path = PosixPath('/Users/teacher/.codex/orchestrator/agent-runtime/cursor/tmp/pytest-of-teacher/pytest-573/test_single_run_bundle_contain0')

    def test_single_run_bundle_contains_the_completed_artifacts(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The opt-in bundle records outputs after the normal run has completed."""
        monkeypatch.chdir(tmp_path)
        config, index, output = _write_single_run_config(tmp_path)
        bundle_dir = tmp_path / "bundle"
        monkeypatch.setattr("pa_core.cli.print_enhanced_summary", lambda *_args, **_kwargs: None)
        _stub_sensitivity(monkeypatch)
    
        main(_single_run_argv(config, index, output, "--bundle", str(bundle_dir)))
    
        meta_path = bundle_dir / RunArtifactBundle._META_FILE
        assert meta_path.is_file()
        bundle_meta = json.loads(meta_path.read_text())
        assert bundle_meta["config_path"] == RunArtifactBundle._CONFIG_FILE
        assert bundle_meta["manifest_path"] == RunArtifactBundle._MANIFEST_FILE
        assert bundle_meta["seed"] == 123
        assert bundle_meta["index_hash"]
    
        outputs_meta = bundle_meta["outputs"]
        assert isinstance(outputs_meta, dict) and outputs_meta
        assert output.name in outputs_meta
    
        assert (bundle_dir / bundle_meta["config_path"]).is_file()
        assert (bundle_dir / bundle_meta["manifest_path"]).is_file()
    
        output_hashes = bundle_meta["hashes"]["outputs"]
        for name, relpath in outputs_meta.items():
            artifact_path = bundle_dir / relpath
            assert artifact_path.is_file(), name
            assert output_hashes.get(name)
    
        loaded = RunArtifactBundle.load(bundle_dir)
>       assert loaded.verify()
E       assert False
E        +  where False = verify()
E        +    where verify = <pa_core.run_artifact_bundle.RunArtifactBundle object at 0x11a017250>.verify

/Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:252: AssertionError
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
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:228: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--bundle", str(bundle_dir)))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts
1 failed, 15 warnings in 0.98s
exit=1
```

```text
$ .venv/bin/python -m pytest -q tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts  # restored
.                                                                        [100%]
=============================== warnings summary ===============================
tests/test_coverage_round_2375.py::test_single_run_bundle_contains_the_completed_artifacts
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/tests/test_coverage_round_2375.py:228: DeprecationWarning: Direct use of pa_core.cli is deprecated; use `pa run` instead.
    main(_single_run_argv(config, index, output, "--bundle", str(bundle_dir)))

tests/test_coverage_round_2375.py: 14 warnings
  /Users/teacher/.codex/orchestrator/worktrees/stranske__Portable-Alpha-Extension-Model__2375__opener/pa_core/config.py:916: DeprecationWarning: Ambiguous return inputs detected for legacy fields: mu_E, mu_H, mu_M, sigma_E, sigma_H, sigma_M. Use explicit keys like mu_H_annual or mu_H_monthly. Legacy field names will be deprecated.
    data = transform(data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1 passed, 15 warnings in 0.91s
exit=0
```
