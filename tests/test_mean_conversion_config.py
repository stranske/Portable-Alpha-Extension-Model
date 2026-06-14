"""Tests for the configurable annual->monthly mean conversion (issue #1912).

The historical default (``simple``, ``mean/12``) does not reproduce the
configured annual mean once monthly returns compound. ``ModelConfig`` now
exposes a ``mean_conversion`` field so callers can opt into the geometric
conversion ``(1+r)**(1/12)-1``, which round-trips back to the annual mean.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from pa_core.config import (
    DEFAULT_MEAN_CONVERSION,
    MONTHS_PER_YEAR,
    ModelConfig,
    annual_mean_to_monthly,
)


def _cfg(**overrides):
    base = dict(
        N_SIMULATIONS=1,
        N_MONTHS=1,
        financing_mode="broadcast",
        mu_H=0.12,
        sigma_H=0.24,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_default_mean_conversion_is_simple_and_backward_compatible() -> None:
    cfg = _cfg()
    assert cfg.mean_conversion == DEFAULT_MEAN_CONVERSION == "simple"
    # Default behaviour unchanged: simple mean/12 conversion.
    assert np.isclose(cfg.mu_H, 0.12 / MONTHS_PER_YEAR)
    assert np.isclose(cfg.mu_H, annual_mean_to_monthly(0.12, method="simple"))


def test_geometric_mean_conversion_applied_to_annual_inputs() -> None:
    cfg = _cfg(mean_conversion="geometric")
    assert cfg.mean_conversion == "geometric"
    expected = (1.0 + 0.12) ** (1.0 / MONTHS_PER_YEAR) - 1.0
    assert np.isclose(cfg.mu_H, expected)
    assert np.isclose(cfg.mu_H, annual_mean_to_monthly(0.12, method="geometric"))


def test_geometric_conversion_reproduces_annual_mean_after_compounding() -> None:
    annual = 0.12
    cfg = _cfg(mu_H=annual, mu_E=annual, mu_M=annual, mean_conversion="geometric")
    # 12 compounded monthly means recover the annual mean (the bug the simple
    # default could not satisfy).
    compounded = (1.0 + cfg.mu_H) ** MONTHS_PER_YEAR - 1.0
    assert np.isclose(compounded, annual)
    # The simple default does NOT recover it -- guards against a silent regression.
    simple_compounded = (1.0 + annual / MONTHS_PER_YEAR) ** MONTHS_PER_YEAR - 1.0
    assert not np.isclose(simple_compounded, annual)


def test_geometric_conversion_applies_to_all_mu_fields() -> None:
    cfg = _cfg(mu_H=0.12, mu_E=0.06, mu_M=0.03, mean_conversion="geometric")
    assert np.isclose(cfg.mu_H, annual_mean_to_monthly(0.12, method="geometric"))
    assert np.isclose(cfg.mu_E, annual_mean_to_monthly(0.06, method="geometric"))
    assert np.isclose(cfg.mu_M, annual_mean_to_monthly(0.03, method="geometric"))


def test_mean_conversion_does_not_affect_volatility() -> None:
    simple = _cfg(sigma_H=0.24, mean_conversion="simple")
    geometric = _cfg(sigma_H=0.24, mean_conversion="geometric")
    # Volatility conversion is independent of the mean-conversion method.
    assert np.isclose(simple.sigma_H, geometric.sigma_H)


def test_mean_conversion_alias_accepted() -> None:
    cfg = _cfg(**{"Mean conversion": "geometric"})
    assert cfg.mean_conversion == "geometric"
    assert np.isclose(cfg.mu_H, annual_mean_to_monthly(0.12, method="geometric"))


def test_monthly_inputs_ignore_mean_conversion() -> None:
    cfg = _cfg(return_unit="monthly", mu_H=0.01, mean_conversion="geometric")
    # No annual->monthly conversion happens for monthly inputs.
    assert np.isclose(cfg.mu_H, 0.01)


def test_invalid_mean_conversion_rejected() -> None:
    with pytest.raises((ValidationError, ValueError)):
        _cfg(mean_conversion="compound")


def test_geometric_config_round_trips() -> None:
    cfg = _cfg(mean_conversion="geometric")
    round_trip = ModelConfig.model_validate(cfg.model_dump())
    assert round_trip.mean_conversion == "geometric"
    # return_unit is already "monthly" after the first conversion, so the mean
    # must not be converted a second time.
    assert np.isclose(round_trip.mu_H, cfg.mu_H)
