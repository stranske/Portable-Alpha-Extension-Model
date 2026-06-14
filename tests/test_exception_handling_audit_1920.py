"""Regression tests for issue #1920.

The 2026-06 audit flagged broad ``except Exception: pass`` blocks that silently
hid failures, and ``assert`` statements used as pre-condition guards (which are
stripped under ``python -O``). These tests pin the corrected behaviour:

- best-effort telemetry failures are *logged* (warning + traceback) rather than
  swallowed silently; and
- degenerate sweep-parameter states raise an explicit ``ValueError`` rather than
  an ``AssertionError`` (or nothing, under ``-O``).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pa_core import sweep
from pa_core.config import SweepConfig, SweepParameter
from pa_core.llm import scenario_fleet


def test_record_scenario_run_logs_instead_of_swallowing(monkeypatch, caplog):
    """Telemetry failures must not propagate, but must be logged (not silent)."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("telemetry backend exploded")

    monkeypatch.setattr(scenario_fleet, "record_fleet_event", _boom)

    config = SimpleNamespace(model_dump=lambda: {"N_SIMULATIONS": 10})

    with caplog.at_level(logging.WARNING, logger=scenario_fleet.__name__):
        # Must NOT raise — the run path is protected.
        scenario_fleet.record_scenario_run(config, None, seed=1)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a warning log when telemetry recording fails"
    assert any("scenario fleet event" in r.getMessage() for r in warnings)
    # exc_info=True means the traceback is attached.
    assert any(r.exc_info is not None for r in warnings)


def test_sample_sweep_value_raises_value_error_without_bounds():
    """A bounds-less parameter is a ValueError, not an AssertionError."""

    rng = np.random.default_rng(0)

    # Stepped sampling path: step present but min/max missing.
    stepped = SweepParameter.model_construct(values=None, min=None, max=None, step=0.1)
    with pytest.raises(ValueError):
        sweep._sample_sweep_value(stepped, rng)

    # Uniform sampling path: no values, no step, no bounds.
    uniform = SweepParameter.model_construct(values=None, min=None, max=None, step=None)
    with pytest.raises(ValueError):
        sweep._sample_sweep_value(uniform, rng)


def test_iter_sweep_grid_raises_value_error_without_step():
    """Grid iteration over a step-less, value-less parameter is a ValueError."""

    param = SweepParameter.model_construct(values=None, min=0.0, max=1.0, step=None)
    cfg = SweepConfig.model_construct(
        method="grid", parameters={"x": param}, samples=None, seed=None
    )
    with pytest.raises(ValueError):
        list(sweep._iter_sweep_grid(cfg))


def test_estimate_total_combinations_raises_value_error_without_step():
    """Counting combinations for a degenerate sweep parameter is a ValueError."""

    param = SweepParameter.model_construct(values=None, min=0.0, max=1.0, step=None)
    cfg = SweepConfig.model_construct(
        method="grid", parameters={"x": param}, samples=None, seed=None
    )
    model_cfg = SimpleNamespace(sweep=cfg)
    with pytest.raises(ValueError):
        sweep._estimate_total_combinations(model_cfg)  # type: ignore[arg-type]


def test_simulate_financing_uses_spawned_rng_without_redundant_guard(monkeypatch):
    """simulate_financing relies on spawn_rngs to return a usable generator."""

    from pa_core.sim import financing

    rng = np.random.default_rng(123)
    monkeypatch.setattr(financing, "spawn_rngs", lambda *a, **kw: [rng])

    result = financing.simulate_financing(12, 0.0, 0.01, 0.0, 1.0)

    assert result.shape == (12,)


def test_facade_agent_semantics_dict_logs_warning_with_exc_info(caplog):
    """_serialize_agent_semantics_input logs warning+traceback for an uncoercible dict."""

    from pa_core.facade import _serialize_agent_semantics_input

    # Unequal-length value lists cannot be passed to pd.DataFrame(), triggering ValueError.
    inputs = {"_agent_semantics_df": {"col_a": [1, 2], "col_b": [3]}}
    with caplog.at_level(logging.WARNING, logger="pa_core.facade"):
        _serialize_agent_semantics_input(inputs)

    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "expected a warning log for bad _agent_semantics_df dict"
    assert any(r.exc_info is not None for r in records)


def test_excel_coerce_agent_semantics_df_dict_logs_warning_with_exc_info(caplog):
    """_coerce_agent_semantics_df logs warning+traceback and returns None for an uncoercible dict."""

    from pa_core.reporting.excel import _coerce_agent_semantics_df

    # Unequal-length value lists cannot be passed to pd.DataFrame(), triggering ValueError.
    with caplog.at_level(logging.WARNING, logger="pa_core.reporting.excel"):
        result = _coerce_agent_semantics_df({"col_a": [1, 2], "col_b": [3]})

    assert result is None
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "expected a warning log for bad dict coercion"
    assert any(r.exc_info is not None for r in records)


def test_excel_tornado_embed_logs_malformed_fallback_without_aborting(tmp_path, caplog):
    """Malformed sensitivity sheet data remains a non-fatal optional chart failure."""

    from pa_core.reporting.excel import export_to_excel

    summary = pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})
    sens_df = pd.DataFrame({"Parameter": ["mu_H"]})
    out_path = tmp_path / "malformed_tornado.xlsx"

    with caplog.at_level(logging.WARNING, logger="pa_core.reporting.excel"):
        export_to_excel({"_sensitivity_df": sens_df}, summary, {}, filename=str(out_path))

    assert out_path.exists()
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("tornado chart" in r.getMessage() for r in records)
    assert any(r.exc_info is not None for r in records)


def test_excel_sunburst_embed_logs_type_error_without_aborting(
    monkeypatch, tmp_path, caplog
):
    """Optional sunburst image failures stay non-fatal even for malformed return data."""

    from pa_core.reporting import excel
    from pa_core.viz import sunburst

    def _raise_type_error(_df):
        raise TypeError("cannot convert missing return")

    summary = pd.DataFrame({"Agent": ["Base"], "terminal_AnnReturn": [0.05]})
    attr_df = pd.DataFrame({"Agent": ["Base"], "Sub": ["Core"], "Return": [None]})
    out_path = tmp_path / "malformed_sunburst.xlsx"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(sunburst, "make", _raise_type_error)

    with caplog.at_level(logging.WARNING, logger=excel.__name__):
        excel.export_to_excel({"_attribution_df": attr_df}, summary, {}, filename=str(out_path))

    assert out_path.exists()
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("sunburst chart" in r.getMessage() for r in records)
    assert any(r.exc_info is not None for r in records)


def test_scenario_fleet_config_mapping_logs_warning_on_model_dump_error(caplog):
    """_config_mapping logs warning+traceback and returns None when model_dump raises."""

    from pa_core.llm.scenario_fleet import _config_mapping

    class _BrokenConfig:
        def model_dump(self) -> None:
            raise RuntimeError("dump failed")

    with caplog.at_level(logging.WARNING, logger=scenario_fleet.__name__):
        result = _config_mapping(_BrokenConfig())

    assert result is None
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "expected a warning log when model_dump() raises"
    assert any(r.exc_info is not None for r in records)


def test_result_explain_fleet_event_logs_instead_of_propagating(monkeypatch, caplog):
    """explain_results_details logs fleet-recording failures rather than propagating them."""

    import pandas as pd

    from pa_core.llm import result_explain

    def _boom(*_args, **_kwargs):
        raise RuntimeError("fleet backend gone")

    monkeypatch.setattr(result_explain, "record_fleet_event", _boom)

    details_df = pd.DataFrame({"Agent": ["A"], "terminal_AnnReturn": [0.05]})
    with caplog.at_level(logging.WARNING, logger=result_explain.__name__):
        text, _trace_url, _payload = result_explain.explain_results_details(details_df)

    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "expected a warning log when fleet recording fails in result_explain"
    assert any(r.exc_info is not None for r in records)
    assert isinstance(text, str)
