from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Mapping, Optional, Sequence, cast

import numpy.typing as npt
from numpy.typing import NDArray

from ..backend import xp as np
from ..random import spawn_rngs
from ..types import GeneratorLike
from ..validators import NUMERICAL_STABILITY_EPSILON
from .params import CANONICAL_PARAMS_MARKER, CANONICAL_PARAMS_VERSION
from .financing import draw_financing_series, simulate_financing

__all__ = [
    "simulate_financing",
    "prepare_mc_universe",
    "prepare_return_shocks",
    "draw_returns",
    "draw_joint_returns",
    "draw_financing_series",
    "simulate_alpha_streams",
]


_VALID_RETURN_DISTS = {"normal", "student_t"}
_VALID_RETURN_COPULAS = {"gaussian", "t"}
_CORR_VALIDATION_TOL = 1e-8

logger = logging.getLogger(__name__)

def _validate_return_draw_settings(
    distribution: str | Sequence[str], copula: str, t_df: float
) -> None:
    if isinstance(distribution, str):
        distributions: tuple[str, ...] = (distribution,)
    else:
        distributions = tuple(distribution)
    for dist in distributions:
        if dist not in _VALID_RETURN_DISTS:
            raise ValueError(f"return_distribution must be one of: {sorted(_VALID_RETURN_DISTS)}")
    if copula not in _VALID_RETURN_COPULAS:
        raise ValueError(f"return_copula must be one of: {sorted(_VALID_RETURN_COPULAS)}")
    if all(dist == "normal" for dist in distributions) and copula != "gaussian":
        raise ValueError("return_copula must be 'gaussian' when return_distribution is 'normal'")
    if any(dist == "student_t" for dist in distributions) and t_df <= 2.0:
        raise ValueError("return_t_df must be greater than 2 for finite variance")


def _resolve_return_distributions(
    base: str, overrides: Optional[Sequence[Optional[str]]] = None
) -> tuple[str, str, str, str]:
    if overrides is None:
        return (base, base, base, base)
    if len(overrides) != 4:
        raise ValueError("return_distributions must have length 4")
    return (
        overrides[0] or base,
        overrides[1] or base,
        overrides[2] or base,
        overrides[3] or base,
    )


def _validate_correlation_matrix(corr: NDArray[Any]) -> None:
    if not np.all(np.isfinite(corr)):
        raise ValueError("Correlation matrix contains non-finite values")
    min_val = float(np.min(corr))
    max_val = float(np.max(corr))
    if min_val < -1.0 - _CORR_VALIDATION_TOL or max_val > 1.0 + _CORR_VALIDATION_TOL:
        raise ValueError(
            "Correlation matrix values must be within [-1, 1]"
            f"; min={min_val:.3f}, max={max_val:.3f}"
        )
    diag = np.diag(corr)
    if not np.allclose(diag, 1.0, atol=_CORR_VALIDATION_TOL):
        idx = int(np.argmax(np.abs(diag - 1.0)))
        raise ValueError(f"Correlation matrix diagonal must be 1; idx {idx} has {diag[idx]:.6f}")


def _project_to_near_psd_correlation(corr: NDArray[Any]) -> NDArray[Any]:
    sym = 0.5 * (corr + corr.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    psd = 0.5 * (psd + psd.T)
    diag = np.sqrt(np.clip(np.diag(psd), 0.0, None))
    if np.any(diag <= 0.0):
        raise ValueError("Correlation repair failed; non-positive diagonal after projection")
    denom = diag[:, None] * diag[None, :]
    repaired = np.divide(psd, denom, out=np.eye(psd.shape[0]), where=denom != 0.0)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return cast(NDArray[Any], repaired)


def _resolve_correlation_matrix(params: Dict[str, Any]) -> tuple[NDArray[Any], dict[str, Any]]:
    corr = np.array(
        [
            [1.0, params["rho_idx_H"], params["rho_idx_E"], params["rho_idx_M"]],
            [params["rho_idx_H"], 1.0, params["rho_H_E"], params["rho_H_M"]],
            [params["rho_idx_E"], params["rho_H_E"], 1.0, params["rho_E_M"]],
            [params["rho_idx_M"], params["rho_H_M"], params["rho_E_M"], 1.0],
        ],
        dtype=float,
    )
    corr = 0.5 * (corr + corr.T)
    _validate_correlation_matrix(corr)

    repair_mode = params.get("correlation_repair_mode", "warn_fix")
    if repair_mode not in {"error", "warn_fix"}:
        raise ValueError("correlation_repair_mode must be 'error' or 'warn_fix'")
    shrinkage = float(params.get("correlation_repair_shrinkage", 0.0) or 0.0)
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("correlation_repair_shrinkage must be between 0 and 1")

    corr_work = corr
    applied_steps: list[str] = []
    if shrinkage > 0.0:
        corr_work = (1.0 - shrinkage) * corr_work + shrinkage * np.eye(corr.shape[0])
        applied_steps.append("shrinkage")

    eigvals_before = np.linalg.eigvalsh(corr_work)
    min_eig_before = float(eigvals_before.min())
    repaired = corr_work
    if min_eig_before < -_CORR_VALIDATION_TOL:
        if repair_mode == "error":
            raise ValueError(
                "Correlation matrix is not PSD; " f"min eigenvalue {min_eig_before:.3e}."
            )
        repaired = _project_to_near_psd_correlation(corr_work)
        applied_steps.append("eigen_clip")
    eigvals_after = np.linalg.eigvalsh(repaired)
    min_eig_after = float(eigvals_after.min())
    max_delta = float(np.max(np.abs(repaired - corr)))
    repair_applied = bool(applied_steps)

    info = {
        "repair_applied": repair_applied,
        "repair_mode": repair_mode,
        "method": "none" if not applied_steps else "+".join(applied_steps),
        "shrinkage": shrinkage,
        "min_eigenvalue_before": min_eig_before,
        "min_eigenvalue_after": min_eig_after,
        "max_abs_delta": max_delta,
    }
    params["_correlation_repair_info"] = info

    if repair_applied:
        logger.warning(
            "Correlation repair applied: mode=%s, method=%s, shrinkage=%.3f, "
            "min_eig_before=%.3e, min_eig_after=%.3e, max_abs_delta=%.3e",
            repair_mode,
            info["method"],
            shrinkage,
            min_eig_before,
            min_eig_after,
            max_delta,
        )
    return repaired, info


def _safe_multivariate_normal(
    rng: GeneratorLike,
    mean: npt.NDArray[Any],
    cov: npt.NDArray[Any],
    size: tuple[int, int],
) -> npt.NDArray[Any]:
    try:
        return cast(npt.NDArray[Any], rng.multivariate_normal(mean=mean, cov=cov, size=size))
    except np.linalg.LinAlgError:
        return cast(
            npt.NDArray[Any],
            rng.multivariate_normal(
                mean=mean,
                cov=cov + np.eye(len(mean)) * NUMERICAL_STABILITY_EPSILON,
                size=size,
            ),
        )


def _draw_student_t(
    *,
    rng: GeneratorLike,
    mean: npt.NDArray[Any],
    sigma: npt.NDArray[Any],
    corr: npt.NDArray[Any],
    size: tuple[int, int],
    df: float,
    copula: str,
) -> npt.NDArray[Any]:
    n_dim = mean.size
    z = _safe_multivariate_normal(rng, np.zeros(n_dim), corr, size)
    scale = np.sqrt((df - 2.0) / df)
    if copula == "t":
        chi = rng.chisquare(df, size=size)
        denom = np.sqrt(chi / df)[..., None]
    else:
        chi = rng.chisquare(df, size=(*size, n_dim))
        denom = np.sqrt(chi / df)
    shocks = z * (scale / denom)
    return cast(npt.NDArray[Any], mean + shocks * sigma)


def _draw_mixed_returns(
    *,
    rng: GeneratorLike,
    mean: npt.NDArray[Any],
    sigma: npt.NDArray[Any],
    corr: npt.NDArray[Any],
    size: tuple[int, int],
    df: float,
    copula: str,
    distributions: Sequence[str],
) -> npt.NDArray[Any]:
    n_dim = mean.size
    z = _safe_multivariate_normal(rng, np.zeros(n_dim), corr, size)
    shocks = np.empty_like(z)
    scale = np.sqrt((df - 2.0) / df)
    denom_common = None
    if copula == "t":
        chi = rng.chisquare(df, size=size)
        denom_common = np.sqrt(chi / df)
    for i, dist in enumerate(distributions):
        if dist == "normal":
            shocks[..., i] = z[..., i]
        else:
            if copula == "t":
                denom = denom_common
            else:
                chi = rng.chisquare(df, size=size)
                denom = np.sqrt(chi / df)
            shocks[..., i] = z[..., i] * (scale / denom)
    return cast(npt.NDArray[Any], mean + shocks * sigma)


def _assert_canonical_params(params: Mapping[str, Any]) -> None:
    marker = params.get(CANONICAL_PARAMS_MARKER)
    if marker != CANONICAL_PARAMS_VERSION:
        raise ValueError("params must be created by build_simulation_params()")

def prepare_mc_universe(
    *,
    N_SIMULATIONS: int,
    N_MONTHS: int,
    mu_idx: float,
    mu_H: float,
    mu_E: float,
    mu_M: float,
    cov_mat: npt.NDArray[Any],
    return_distribution: str = "normal",
    return_t_df: float = 5.0,
    return_copula: str = "gaussian",
    return_distributions: Optional[Sequence[Optional[str]]] = None,
    seed: Optional[int] = None,
    rng: Optional[GeneratorLike] = None,
) -> npt.NDArray[Any]:
    """Return stacked draws of (index, H, E, M) returns."""
    warnings.warn(
        "prepare_mc_universe is deprecated; use draw_joint_returns instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if N_SIMULATIONS <= 0 or N_MONTHS <= 0:
        raise ValueError("N_SIMULATIONS and N_MONTHS must be positive")
    if cov_mat.shape != (4, 4):
        raise ValueError("cov_mat must be 4×4 and ordered as [idx, H, E, M]")
    if rng is None:
        rng = spawn_rngs(seed, 1)[0]
    assert rng is not None
    distributions = _resolve_return_distributions(return_distribution, return_distributions)
    _validate_return_draw_settings(distributions, return_copula, return_t_df)
    mean = np.array([mu_idx, mu_H, mu_E, mu_M])
    cov = cov_mat
    if all(dist == "normal" for dist in distributions):
        sims = _safe_multivariate_normal(rng, mean, cov, (N_SIMULATIONS, N_MONTHS))
    else:
        sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        denom = np.outer(sigma, sigma)
        corr = np.divide(
            cov,
            denom,
            out=np.eye(cov.shape[0]),
            where=denom != 0.0,
        )
        if all(dist == "student_t" for dist in distributions):
            sims = _draw_student_t(
                rng=rng,
                mean=mean,
                sigma=sigma,
                corr=corr,
                size=(N_SIMULATIONS, N_MONTHS),
                df=return_t_df,
                copula=return_copula,
            )
        else:
            sims = _draw_mixed_returns(
                rng=rng,
                mean=mean,
                sigma=sigma,
                corr=corr,
                size=(N_SIMULATIONS, N_MONTHS),
                df=return_t_df,
                copula=return_copula,
                distributions=distributions,
            )
    return sims


def prepare_return_shocks(
    *,
    n_months: int,
    n_sim: int,
    params: Dict[str, Any],
    rng: Optional[GeneratorLike] = None,
) -> Dict[str, Any]:
    """Pre-generate return shocks to reuse across parameter combinations."""
    if rng is None:
        rng = spawn_rngs(None, 1)[0]
    assert rng is not None
    distribution = params.get("return_distribution", "normal")
    dist_overrides = (
        params.get("return_distribution_idx"),
        params.get("return_distribution_H"),
        params.get("return_distribution_E"),
        params.get("return_distribution_M"),
    )
    use_overrides = any(val is not None for val in dist_overrides)
    copula = params.get("return_copula", "gaussian")
    t_df = float(params.get("return_t_df", 5.0))
    distributions = _resolve_return_distributions(
        distribution, dist_overrides if use_overrides else None
    )
    _validate_return_draw_settings(distributions, copula, t_df)
    corr, repair_info = _resolve_correlation_matrix(params)
    z = _safe_multivariate_normal(rng, np.zeros(4), corr, (n_sim, n_months))
    shocks: Dict[str, Any] = {
        "z": z,
        "distributions": distributions,
        "copula": copula,
        "t_df": t_df,
        "corr": corr,
        "corr_repair_info": repair_info,
    }
    if any(dist == "student_t" for dist in distributions):
        if copula == "t":
            shocks["chi_common"] = rng.chisquare(t_df, size=(n_sim, n_months))
        else:
            shocks["chi_dim"] = rng.chisquare(t_df, size=(n_sim, n_months, 4))
    return shocks


def draw_returns(
    *,
    n_months: int,
    n_sim: int,
    params: Dict[str, Any],
    rng: Optional[GeneratorLike] = None,
    shocks: Optional[Dict[str, Any]] = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Vectorised draw of monthly returns for (beta, H, E, M)."""
    _assert_canonical_params(params)
    distribution = params.get("return_distribution", "normal")
    dist_overrides = (
        params.get("return_distribution_idx"),
        params.get("return_distribution_H"),
        params.get("return_distribution_E"),
        params.get("return_distribution_M"),
    )
    use_overrides = any(val is not None for val in dist_overrides)
    copula = params.get("return_copula", "gaussian")
    t_df = float(params.get("return_t_df", 5.0))
    distributions = _resolve_return_distributions(
        distribution, dist_overrides if use_overrides else None
    )
    _validate_return_draw_settings(distributions, copula, t_df)
    μ_idx = params["mu_idx_month"]
    μ_H = params["default_mu_H"]
    μ_E = params["default_mu_E"]
    μ_M = params["default_mu_M"]
    σ_idx = params["idx_sigma_month"]
    σ_H = params["default_sigma_H"]
    σ_E = params["default_sigma_E"]
    σ_M = params["default_sigma_M"]
    corr, _ = _resolve_correlation_matrix(params)
    μ = np.array([μ_idx, μ_H, μ_E, μ_M])
    σ = np.array([σ_idx, σ_H, σ_E, σ_M])
    if shocks is not None:
        if (
            shocks.get("distributions") != distributions
            or shocks.get("copula") != copula
            or float(shocks.get("t_df", t_df)) != t_df
            or not np.allclose(shocks.get("corr"), corr)
        ):
            raise ValueError("Return shocks are not compatible with current parameters")
        z = shocks["z"]
        if z.shape != (n_sim, n_months, 4):
            raise ValueError("Return shocks have incompatible shape")
        if all(dist == "normal" for dist in distributions):
            sims = μ + z * σ
        else:
            scale = np.sqrt((t_df - 2.0) / t_df)
            if all(dist == "student_t" for dist in distributions):
                if copula == "t":
                    chi = shocks.get("chi_common")
                    if chi is None:
                        raise ValueError("Missing chi_common for t copula shocks")
                    denom = np.sqrt(chi / t_df)[..., None]
                else:
                    chi = shocks.get("chi_dim")
                    if chi is None:
                        raise ValueError("Missing chi_dim for gaussian copula shocks")
                    denom = np.sqrt(chi / t_df)
                sims = μ + (z * (scale / denom)) * σ
            else:
                shocks_out = np.empty_like(z)
                denom_common = None
                if copula == "t":
                    chi = shocks.get("chi_common")
                    if chi is None:
                        raise ValueError("Missing chi_common for t copula shocks")
                    denom_common = np.sqrt(chi / t_df)
                for i, dist in enumerate(distributions):
                    if dist == "normal":
                        shocks_out[..., i] = z[..., i]
                    else:
                        if copula == "t":
                            denom = denom_common
                        else:
                            chi = shocks.get("chi_dim")
                            if chi is None:
                                raise ValueError("Missing chi_dim for gaussian copula shocks")
                            denom = np.sqrt(chi[..., i] / t_df)
                        shocks_out[..., i] = z[..., i] * (scale / denom)
                sims = μ + shocks_out * σ
    else:
        if rng is None:
            rng = spawn_rngs(None, 1)[0]
        assert rng is not None
        if all(dist == "normal" for dist in distributions):
            Σ = corr * (σ[:, None] * σ[None, :])
            sims = _safe_multivariate_normal(rng, μ, Σ, (n_sim, n_months))
        else:
            if all(dist == "student_t" for dist in distributions):
                sims = _draw_student_t(
                    rng=rng,
                    mean=μ,
                    sigma=σ,
                    corr=corr,
                    size=(n_sim, n_months),
                    df=t_df,
                    copula=copula,
                )
            else:
                sims = _draw_mixed_returns(
                    rng=rng,
                    mean=μ,
                    sigma=σ,
                    corr=corr,
                    size=(n_sim, n_months),
                    df=t_df,
                    copula=copula,
                    distributions=distributions,
                )
    r_beta = sims[:, :, 0]
    r_H = sims[:, :, 1]
    r_E = sims[:, :, 2]
    r_M = sims[:, :, 3]
    return r_beta, r_H, r_E, r_M


def draw_joint_returns(
    *,
    n_months: int,
    n_sim: int,
    params: Dict[str, Any],
    rng: Optional[GeneratorLike] = None,
    shocks: Optional[Dict[str, Any]] = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Backward-compatible wrapper for draw_returns."""
    return draw_returns(
        n_months=n_months,
        n_sim=n_sim,
        params=params,
        rng=rng,
        shocks=shocks,
    )


def simulate_alpha_streams(
    T: int,
    cov: npt.NDArray[Any],
    mu_idx: float,
    mu_H: float,
    mu_E: float,
    mu_M: float,
    *,
    return_distribution: str = "normal",
    return_t_df: float = 5.0,
    return_copula: str = "gaussian",
    return_distributions: Optional[Sequence[Optional[str]]] = None,
    rng: Optional[GeneratorLike] = None,
) -> NDArray[Any]:
    """Simulate T observations of (Index_return, H, E, M)."""
    if T <= 0:
        raise ValueError("T must be positive")
    if cov.shape != (4, 4):
        raise ValueError("cov must be 4×4 and ordered as [idx, H, E, M]")
    distributions = _resolve_return_distributions(return_distribution, return_distributions)
    _validate_return_draw_settings(distributions, return_copula, return_t_df)
    means = np.array([mu_idx, mu_H, mu_E, mu_M])
    if rng is None:
        rng = spawn_rngs(None, 1)[0]
    assert rng is not None
    if all(dist == "normal" for dist in distributions):
        return _safe_multivariate_normal(rng, means, cov, (T, 1))[:, 0, :]
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(sigma, sigma)
    corr = np.divide(
        cov,
        denom,
        out=np.eye(cov.shape[0]),
        where=denom != 0.0,
    )
    if all(dist == "student_t" for dist in distributions):
        sims = _draw_student_t(
            rng=rng,
            mean=means,
            sigma=sigma,
            corr=corr,
            size=(T, 1),
            df=return_t_df,
            copula=return_copula,
        )
    else:
        sims = _draw_mixed_returns(
            rng=rng,
            mean=means,
            sigma=sigma,
            corr=corr,
            size=(T, 1),
            df=return_t_df,
            copula=return_copula,
            distributions=distributions,
        )
    return sims[:, 0, :]
