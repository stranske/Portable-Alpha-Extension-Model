from __future__ import annotations

import warnings

import numpy as np
import pytest

from pa_core.sim import (
    build_cov_matrix,
    build_generic_cov_matrix,
    draw_named_returns,
    map_sleeve_alpha_streams,
    prepare_mc_universe,
    simulate_alpha_streams,
)


def _rank_deficient_float32_covariance() -> np.ndarray:
    factors = np.random.default_rng(0).standard_normal((4, 2)).astype(np.float32)
    return factors @ factors.T


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
    stream_names = (
        "idx",
        "H",
        "E",
        "M",
        "portfolio:trend_growth",
        "portfolio:trend_value",
        "stream:trend_core",
    )
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
            "stream:trend_core": 0.07,
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
            "prefixed_stream_manager": "portfolio:trend_core",
        },
        draws,
    )

    assert set(draws) == set(stream_names)
    assert draws["portfolio:trend_growth"].shape == (4, 3)
    np.testing.assert_array_equal(sleeve_streams["growth_manager"], draws["portfolio:trend_growth"])
    np.testing.assert_array_equal(sleeve_streams["value_manager"], draws["portfolio:trend_value"])
    np.testing.assert_array_equal(
        sleeve_streams["prefixed_stream_manager"], draws["stream:trend_core"]
    )


def test_named_return_draws_reject_nonfinite_means() -> None:
    with pytest.raises(ValueError, match="means must contain only finite"):
        draw_named_returns(
            n_months=2,
            n_sim=2,
            stream_names=("idx", "alpha"),
            means=(float("nan"), 0.0),
            cov=np.eye(2),
            seed=1,
        )


def test_named_return_draws_reject_dimension_amplified_asymmetry() -> None:
    size = 64
    covariance = np.eye(size, dtype=np.float16)
    covariance[0, 1] = np.float16(1.0)
    covariance[1, 0] = np.float16(0.9375)

    with pytest.raises(ValueError, match="symmetric"):
        draw_named_returns(
            n_months=1,
            n_sim=1,
            stream_names=tuple(f"stream_{index}" for index in range(size)),
            means=(0.0,) * size,
            cov=covariance,
            seed=1,
        )


def test_named_return_draws_reject_dimension_amplified_correlation() -> None:
    size = 64
    covariance = np.eye(size, dtype=np.float16)
    covariance[0, 1] = covariance[1, 0] = np.float16(1.05)

    with pytest.raises(ValueError, match="absolute correlation cannot exceed 1"):
        draw_named_returns(
            n_months=1,
            n_sim=1,
            stream_names=tuple(f"stream_{index}" for index in range(size)),
            means=(0.0,) * size,
            cov=covariance,
            seed=1,
        )


@pytest.mark.parametrize(
    ("cov", "message"),
    [
        (np.array([[float("nan"), 0.0], [0.0, 1.0]]), "non-finite"),
        (np.array([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
        (np.array([[1.0e-20, 1.0e-12], [0.0, 1.0e-20]]), "symmetric"),
        (
            np.array([[1.0, 1.0e-7], [0.0, 1.0e-20]], dtype=np.float32),
            "symmetric",
        ),
        (np.array([[-1.0e-9, 0.0], [0.0, 1.0]]), "variances must be non-negative"),
        (np.array([[1.0e-20, 2.0e-20], [2.0e-20, 1.0e-20]]), "positive semidefinite"),
        (
            np.array([[1.0, 5.0e-8], [5.0e-8, 1.0e-20]], dtype=np.float32),
            "positive semidefinite",
        ),
        (np.array([[0.0, 1.0e-20], [1.0e-20, 1.0]]), "positive semidefinite"),
    ],
    ids=[
        "nonfinite",
        "asymmetric",
        "tiny-asymmetric",
        "heterogeneous-asymmetric",
        "negative-variance",
        "scale-sensitive-indefinite",
        "heterogeneous-indefinite",
        "zero-variance-covariance",
    ],
)
def test_named_return_draws_reject_invalid_covariance(cov: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        draw_named_returns(
            n_months=2,
            n_sim=2,
            stream_names=("idx", "alpha"),
            means=(0.0, 0.0),
            cov=cov,
            seed=1,
        )


def test_named_return_draws_accept_rank_deficient_float32_covariance() -> None:
    covariance = _rank_deficient_float32_covariance()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        draws = draw_named_returns(
            n_months=2,
            n_sim=2,
            stream_names=("idx", "alpha_a", "alpha_b", "alpha_c"),
            means=(0.0, 0.0, 0.0, 0.0),
            cov=covariance,
            seed=1,
        )

    assert set(draws) == {"idx", "alpha_a", "alpha_b", "alpha_c"}
    assert all(values.shape == (2, 2) for values in draws.values())
    assert all(np.all(np.isfinite(values)) for values in draws.values())


def test_named_return_draws_accept_large_finite_covariance_without_overflow() -> None:
    covariance = np.diag([0.75 * np.finfo(np.float64).max] * 2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        draws = draw_named_returns(
            n_months=1,
            n_sim=1,
            stream_names=("idx", "alpha"),
            means=(0.0, 0.0),
            cov=covariance,
            seed=1,
        )

    assert all(np.all(np.isfinite(values)) for values in draws.values())


def test_prepare_mc_universe_uses_repaired_rank_deficient_covariance() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("error", RuntimeWarning)
        draws = prepare_mc_universe(
            N_SIMULATIONS=2,
            N_MONTHS=2,
            mu_idx=0.0,
            mu_H=0.0,
            mu_E=0.0,
            mu_M=0.0,
            cov_mat=_rank_deficient_float32_covariance(),
            seed=1,
        )

    assert draws.shape == (2, 2, 4)


def test_simulate_alpha_streams_uses_repaired_rank_deficient_covariance() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        draws = simulate_alpha_streams(
            2,
            _rank_deficient_float32_covariance(),
            0.0,
            0.0,
            0.0,
            0.0,
            seed=1,
        )

    assert draws.shape == (2, 4)


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
