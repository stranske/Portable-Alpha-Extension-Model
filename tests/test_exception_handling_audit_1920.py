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
