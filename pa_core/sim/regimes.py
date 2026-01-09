from __future__ import annotations

from typing import Any, Sequence, cast

import numpy.typing as npt

from ..backend import xp as np
from ..config import ModelConfig
from ..types import GeneratorLike
from .covariance import build_cov_matrix
from .params import build_simulation_params
from .simulation_initialization import ensure_rng


def _cov_to_corr_and_sigma(cov: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(sigma, sigma)
    corr = np.divide(cov, denom, out=np.eye(cov.shape[0]), where=denom != 0.0)
    return sigma, corr


def build_regime_draw_params(
    cfg: ModelConfig,
    *,
    mu_idx: float,
    idx_sigma: float,
    n_samples: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build per-regime simulation parameter dicts from a base config."""
    if cfg.regimes is None:
        raise ValueError("regimes are required to build regime parameters")

    params: list[dict[str, Any]] = []
    labels: list[str] = []
    for regime in cfg.regimes:
        sigma_h = float(regime.sigma_H) if regime.sigma_H is not None else float(cfg.sigma_H)
        sigma_e = float(regime.sigma_E) if regime.sigma_E is not None else float(cfg.sigma_E)
        sigma_m = float(regime.sigma_M) if regime.sigma_M is not None else float(cfg.sigma_M)
        idx_sigma_regime = float(idx_sigma) * float(regime.idx_sigma_multiplier)

        rho_idx_H = (
            float(regime.rho_idx_H) if regime.rho_idx_H is not None else float(cfg.rho_idx_H)
        )
        rho_idx_E = (
            float(regime.rho_idx_E) if regime.rho_idx_E is not None else float(cfg.rho_idx_E)
        )
        rho_idx_M = (
            float(regime.rho_idx_M) if regime.rho_idx_M is not None else float(cfg.rho_idx_M)
        )
        rho_H_E = float(regime.rho_H_E) if regime.rho_H_E is not None else float(cfg.rho_H_E)
        rho_H_M = float(regime.rho_H_M) if regime.rho_H_M is not None else float(cfg.rho_H_M)
        rho_E_M = float(regime.rho_E_M) if regime.rho_E_M is not None else float(cfg.rho_E_M)

        cov = build_cov_matrix(
            rho_idx_H,
            rho_idx_E,
            rho_idx_M,
            rho_H_E,
            rho_H_M,
            rho_E_M,
            idx_sigma_regime,
            sigma_h,
            sigma_e,
            sigma_m,
            covariance_shrinkage=cfg.covariance_shrinkage,
            n_samples=n_samples,
        )
        sigma_vec, corr_mat = _cov_to_corr_and_sigma(cov)
        return_overrides = {
            "default_sigma_H": float(sigma_vec[1]),
            "default_sigma_E": float(sigma_vec[2]),
            "default_sigma_M": float(sigma_vec[3]),
            "rho_idx_H": float(corr_mat[0, 1]),
            "rho_idx_E": float(corr_mat[0, 2]),
            "rho_idx_M": float(corr_mat[0, 3]),
            "rho_H_E": float(corr_mat[1, 2]),
            "rho_H_M": float(corr_mat[1, 3]),
            "rho_E_M": float(corr_mat[2, 3]),
        }
        params.append(
            build_simulation_params(
                cfg,
                mu_idx=mu_idx,
                idx_sigma=float(sigma_vec[0]),
                return_overrides=return_overrides,
            )
        )
        labels.append(regime.name)
    return params, labels


def resolve_regime_start(cfg: ModelConfig) -> int:
    if cfg.regimes is None:
        raise ValueError("regimes are required to resolve regime_start")
    if cfg.regime_start is None:
        return 0
    names = [regime.name for regime in cfg.regimes]
    return names.index(cfg.regime_start)


def simulate_regime_paths(
    *,
    n_sim: int,
    n_months: int,
    transition: Sequence[Sequence[float]],
    start_state: int,
    seed: int | None = None,
    rng: GeneratorLike | None = None,
) -> npt.NDArray[Any]:
    """Simulate regime paths using a Markov transition matrix.

    ``seed`` is used to create a per-run generator when ``rng`` is not supplied.
    """
    if n_sim <= 0 or n_months <= 0:
        raise ValueError("n_sim and n_months must be positive")
    transition_mat = np.asarray(transition, dtype=float)
    if transition_mat.ndim != 2 or transition_mat.shape[0] != transition_mat.shape[1]:
        raise ValueError("transition must be a square matrix")
    n_regimes = int(transition_mat.shape[0])
    if not 0 <= start_state < n_regimes:
        raise ValueError("start_state must be within regime index range")
    rng = ensure_rng(seed, rng)

    paths = np.empty((n_sim, n_months), dtype=int)
    paths[:, 0] = start_state
    for t in range(1, n_months):
        prev = paths[:, t - 1]
        for regime_idx in range(n_regimes):
            mask = prev == regime_idx
            count = int(mask.sum())
            if count == 0:
                continue
            draws = rng.random(size=count)
            cum_probs = np.cumsum(transition_mat[regime_idx])
            paths[mask, t] = np.searchsorted(cum_probs, draws)
    return cast(npt.NDArray[Any], paths)


def apply_regime_labels(paths: npt.NDArray[Any], labels: Sequence[str]) -> npt.NDArray[Any]:
    label_array = np.asarray(labels, dtype=object)
    return cast(npt.NDArray[Any], label_array[paths])


__all__ = [
    "apply_regime_labels",
    "build_regime_draw_params",
    "resolve_regime_start",
    "simulate_regime_paths",
]
