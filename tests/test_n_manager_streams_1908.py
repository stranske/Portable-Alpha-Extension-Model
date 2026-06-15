from __future__ import annotations

import numpy as np
import pytest

from pa_core.sim import (
    build_cov_matrix,
    build_generic_cov_matrix,
    draw_named_returns,
    map_sleeve_alpha_streams,
    simulate_alpha_streams,
)


def test_generic_covariance_supports_manager_universe_shape() -> None:
    stream_names = ("idx", "H", "E", "M", "trend_growth", "trend_value")
    sigmas = {
        "idx": 0.035,
        "H": 0.010,
        "E": 0.018,
        "M": 0.020,
        "trend_growth": 0.025,
        "trend_value": 0.022,
    }
    correlations = {
        (left, right): 0.05
        for i, left in enumerate(stream_names)
        for right in stream_names[i + 1 :]
    }
    correlations[("trend_growth", "trend_value")] = 0.35

    cov = build_generic_cov_matrix(stream_names, sigmas, correlations)

    assert cov.shape == (6, 6)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= -1e-12
    assert cov[4, 5] == pytest.approx(sigmas["trend_growth"] * sigmas["trend_value"] * 0.35)


def test_named_return_draws_map_sleeves_to_alpha_sources() -> None:
    stream_names = ("idx", "H", "E", "M", "portfolio:trend_growth", "portfolio:trend_value")
    correlations = {
        (left, right): 0.0 for i, left in enumerate(stream_names) for right in stream_names[i + 1 :]
    }
    cov = build_generic_cov_matrix(
        stream_names,
        {
            "idx": 0.01,
            "H": 0.02,
            "E": 0.03,
            "M": 0.04,
            "portfolio:trend_growth": 0.05,
            "portfolio:trend_value": 0.06,
        },
        correlations,
    )

    draws = draw_named_returns(
        n_months=3,
        n_sim=4,
        stream_names=stream_names,
        means={name: idx * 0.001 for idx, name in enumerate(stream_names)},
        cov=cov,
        seed=1908,
    )
    sleeve_streams = map_sleeve_alpha_streams(
        {
            "growth_manager": "portfolio:trend_growth",
            "value_manager": "trend_value",
        },
        draws,
    )

    assert set(draws) == set(stream_names)
    assert draws["portfolio:trend_growth"].shape == (4, 3)
    np.testing.assert_array_equal(sleeve_streams["growth_manager"], draws["portfolio:trend_growth"])
    np.testing.assert_array_equal(sleeve_streams["value_manager"], draws["portfolio:trend_value"])


def test_legacy_four_stream_api_remains_compatible() -> None:
    cov = build_cov_matrix(
        0.05,
        0.0,
        0.0,
        0.1,
        0.1,
        0.0,
        0.03,
        0.01,
        0.02,
        0.02,
    )

    draws = simulate_alpha_streams(
        5,
        cov,
        0.001,
        0.002,
        0.003,
        0.004,
        seed=1908,
    )

    assert cov.shape == (4, 4)
    assert draws.shape == (5, 4)
