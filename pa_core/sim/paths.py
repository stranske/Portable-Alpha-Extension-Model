from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, cast

import numpy.typing as npt
from numpy.random import Generator
from numpy.typing import NDArray

from ..backend import xp as np
from ..random import spawn_rngs
from ..validators import NUMERICAL_STABILITY_EPSILON

__all__ = [
    "simulate_financing",
    "prepare_mc_universe",
    "prepare_return_shocks",
    "draw_joint_returns",
    "draw_financing_series",
    "simulate_alpha_streams",
]


_VALID_RETURN_DISTS = {"normal", "student_t"}
_VALID_RETURN_COPULAS = {"gaussian", "t"}


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


def _safe_multivariate_normal(
    rng: Generator,
    mean: npt.NDArray[Any],
    cov: npt.NDArray[Any],
    size: tuple[int, int],
) -> npt.NDArray[Any]:
    try:
        return rng.multivariate_normal(mean=mean, cov=cov, size=size)
    except np.linalg.LinAlgError:
        return rng.multivariate_normal(
            mean=mean,
            cov=cov + np.eye(len(mean)) * NUMERICAL_STABILITY_EPSILON,
            size=size,
        )


def _draw_student_t(
    *,
    rng: Generator,
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
    rng: Generator,
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


def simulate_financing(
    T: int,
    financing_mean: float,
    financing_sigma: float,
    spike_prob: float,
    spike_factor: float,
    *,
    seed: Optional[int] = None,
    n_scenarios: int = 1,
    rng: Optional[Generator] = None,
) -> npt.NDArray[Any]:
    """Vectorised financing spread simulation with optional spikes."""
    if T <= 0:
        raise ValueError("T must be positive")
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    if rng is None:
        rng = spawn_rngs(seed, 1)[0]
    assert rng is not None
    base = rng.normal(loc=financing_mean, scale=financing_sigma, size=(n_scenarios, T))
    jumps = (rng.random(size=(n_scenarios, T)) < spike_prob) * (spike_factor * financing_sigma)
    out = np.clip(base + jumps, 0.0, None)
    return out[0] if n_scenarios == 1 else out  # type: ignore[no-any-return]


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
    rng: Optional[Generator] = None,
) -> npt.NDArray[Any]:
    """Return stacked draws of (index, H, E, M) returns."""
    if N_SIMULATIONS <= 0 or N_MONTHS <= 0:
        raise ValueError("N_SIMULATIONS and N_MONTHS must be positive")
    if cov_mat.shape != (4, 4):
        raise ValueError("cov_mat must be 4×4 and ordered as [idx, H, E, M]")
    if rng is None:
        rng = spawn_rngs(seed, 1)[0]
    assert rng is not None
    distributions = _resolve_return_distributions(return_distribution, return_distributions)
    _validate_return_draw_settings(distributions, return_copula, return_t_df)
    mean = np.array([mu_idx, mu_H, mu_E, mu_M]) / 12.0
    cov = cov_mat / 12.0
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
    rng: Optional[Generator] = None,
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
    ρ_idx_H = params["rho_idx_H"]
    ρ_idx_E = params["rho_idx_E"]
    ρ_idx_M = params["rho_idx_M"]
    ρ_H_E = params["rho_H_E"]
    ρ_H_M = params["rho_H_M"]
    ρ_E_M = params["rho_E_M"]
    corr = np.array(
        [
            [1.0, ρ_idx_H, ρ_idx_E, ρ_idx_M],
            [ρ_idx_H, 1.0, ρ_H_E, ρ_H_M],
            [ρ_idx_E, ρ_H_E, 1.0, ρ_E_M],
            [ρ_idx_M, ρ_H_M, ρ_E_M, 1.0],
        ]
    )
    z = _safe_multivariate_normal(rng, np.zeros(4), corr, (n_sim, n_months))
    shocks: Dict[str, Any] = {
        "z": z,
        "distributions": distributions,
        "copula": copula,
        "t_df": t_df,
        "corr": corr,
    }
    if any(dist == "student_t" for dist in distributions):
        if copula == "t":
            shocks["chi_common"] = rng.chisquare(t_df, size=(n_sim, n_months))
        else:
            shocks["chi_dim"] = rng.chisquare(t_df, size=(n_sim, n_months, 4))
    return shocks


def draw_joint_returns(
    *,
    n_months: int,
    n_sim: int,
    params: Dict[str, Any],
    rng: Optional[Generator] = None,
    shocks: Optional[Dict[str, Any]] = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Vectorised draw of monthly returns for (beta, H, E, M)."""
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
    ρ_idx_H = params["rho_idx_H"]
    ρ_idx_E = params["rho_idx_E"]
    ρ_idx_M = params["rho_idx_M"]
    ρ_H_E = params["rho_H_E"]
    ρ_H_M = params["rho_H_M"]
    ρ_E_M = params["rho_E_M"]
    corr = np.array(
        [
            [1.0, ρ_idx_H, ρ_idx_E, ρ_idx_M],
            [ρ_idx_H, 1.0, ρ_H_E, ρ_H_M],
            [ρ_idx_E, ρ_H_E, 1.0, ρ_E_M],
            [ρ_idx_M, ρ_H_M, ρ_E_M, 1.0],
        ]
    )
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


def draw_financing_series(
    *,
    n_months: int,
    n_sim: int,
    params: Dict[str, Any],
    rng: Optional[Generator] = None,
    rngs: Optional[Mapping[str, Generator]] = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Return three matrices of monthly financing spreads.

    ``rngs`` may provide dedicated generators for each sleeve under the keys
    ``"internal"``, ``"external_pa"``, and ``"active_ext"``. If not supplied,
    ``rng`` will be used for all sleeves.
    """
    if rngs is not None:
        tmp_int = rngs.get("internal")
        if isinstance(tmp_int, Generator):
            r_int: Generator = tmp_int
        else:
            r_int = spawn_rngs(None, 1)[0]

        tmp_ext = rngs.get("external_pa")
        if isinstance(tmp_ext, Generator):
            r_ext: Generator = tmp_ext
        else:
            r_ext = spawn_rngs(None, 1)[0]

        tmp_act = rngs.get("active_ext")
        if isinstance(tmp_act, Generator):
            r_act: Generator = tmp_act
        else:
            r_act = spawn_rngs(None, 1)[0]
    else:
        if rng is None:
            rng = spawn_rngs(None, 1)[0]
        assert isinstance(rng, Generator)
        r_int = rng
        r_ext = rng
        r_act = rng

    def _sim(
        mean_key: str,
        sigma_key: str,
        p_key: str,
        k_key: str,
        rng_local: Generator,
    ) -> npt.NDArray[Any]:
        mean = params[mean_key]
        sigma = params[sigma_key]
        p = params[p_key]
        k = params[k_key]
        vec = simulate_financing(
            n_months,
            mean,
            sigma,
            p,
            k,
            n_scenarios=1,
            rng=rng_local,
        )
        return np.broadcast_to(vec, (n_sim, n_months))  # type: ignore[no-any-return]

    f_int_mat = _sim(
        "internal_financing_mean_month",
        "internal_financing_sigma_month",
        "internal_spike_prob",
        "internal_spike_factor",
        r_int,
    )
    f_ext_pa_mat = _sim(
        "ext_pa_financing_mean_month",
        "ext_pa_financing_sigma_month",
        "ext_pa_spike_prob",
        "ext_pa_spike_factor",
        r_ext,
    )
    f_act_ext_mat = _sim(
        "act_ext_financing_mean_month",
        "act_ext_financing_sigma_month",
        "act_ext_spike_prob",
        "act_ext_spike_factor",
        r_act,
    )
    return f_int_mat, f_ext_pa_mat, f_act_ext_mat


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
    rng: Optional[Generator] = None,
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
        return _safe_multivariate_normal(rng, means, cov, (T, 1))[:, 0, :]  # type: ignore[no-any-return]
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
    return sims[:, 0, :]  # type: ignore[no-any-return]
