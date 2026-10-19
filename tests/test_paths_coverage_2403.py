"""Coverage checks for issue #2403 return-path helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pa_core.sim.paths import (
    _normalize_alpha_source,
    _resolve_return_distributions,
    _resolve_stream_return_distributions,
    _validate_return_draw_settings,
    draw_named_returns,
    map_sleeve_alpha_streams,
)


def test_return_distribution_helpers_validate_and_fill_overrides() -> None:
    assert _resolve_return_distributions("normal") == ("normal",) * 4
    assert _resolve_return_distributions("normal", (None, "student_t", None, None)) == (
        "normal",
        "student_t",
        "normal",
        "normal",
    )
    assert _resolve_stream_return_distributions(
        "normal", ("index", "alpha"), {"alpha": "student_t"}
    ) == ("normal", "student_t")
    assert _validate_return_draw_settings(("normal", "student_t"), "t", "5") == 5.0

    with pytest.raises(ValueError, match="length 4"):
        _resolve_return_distributions("normal", ("normal",))
    with pytest.raises(ValueError, match="same length"):
        _resolve_stream_return_distributions("normal", ("index", "alpha"), ("normal",))
    with pytest.raises(ValueError, match="one of"):
        _validate_return_draw_settings("lognormal", "gaussian", 5)
    with pytest.raises(ValueError, match="gaussian"):
        _validate_return_draw_settings("normal", "t", 5)
    with pytest.raises(ValueError, match="greater than 2"):
        _validate_return_draw_settings("student_t", "t", 2)


def test_named_returns_and_sleeve_mapping_cover_mixed_stream_sources() -> None:
    streams = draw_named_returns(
        n_months=3,
        n_sim=2,
        stream_names=("index", "manager_a", "portfolio:trend"),
        means={"index": 0.0, "manager_a": 0.01, "portfolio:trend": 0.02},
        cov=np.eye(3) * 0.0001,
        return_distribution="normal",
        seed=7,
    )

    mapped = map_sleeve_alpha_streams(
        {"plain": "manager_a", "prefixed": "portfolio:trend", "derived": "trend"}, streams
    )

    assert set(streams) == {"index", "manager_a", "portfolio:trend"}
    assert all(values.shape == (2, 3) for values in streams.values())
    assert mapped["plain"] is streams["manager_a"]
    assert mapped["prefixed"] is streams["portfolio:trend"]
    assert mapped["derived"] is streams["portfolio:trend"]
    assert _normalize_alpha_source(" portfolio:trend ")[0] == "portfolio:trend"
    with pytest.raises(ValueError, match="must not be empty"):
        _normalize_alpha_source("  ")
    with pytest.raises(KeyError, match="does not match"):
        map_sleeve_alpha_streams({"missing": "manager_b"}, streams)


def test_named_returns_rejects_invalid_dimensions_means_and_covariance() -> None:
    kwargs = {
        "n_months": 1,
        "n_sim": 1,
        "stream_names": ("index", "alpha"),
        "means": (0.0, 0.01),
        "cov": np.eye(2),
    }
    with pytest.raises(ValueError, match="must be positive"):
        draw_named_returns(**(kwargs | {"n_months": 0}))
    with pytest.raises(ValueError, match="must be unique"):
        draw_named_returns(**(kwargs | {"stream_names": ("index", "index")}))
    with pytest.raises(ValueError, match="cov shape"):
        draw_named_returns(**(kwargs | {"cov": np.eye(3)}))
    with pytest.raises(ValueError, match="missing mean"):
        draw_named_returns(**(kwargs | {"means": {"index": 0.0}}))
