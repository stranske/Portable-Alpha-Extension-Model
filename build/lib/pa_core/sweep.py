from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

# tqdm is optional; provide a no-op fallback wrapper to avoid hard dependency at import time
try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    _HAS_TQDM = False

from .agents.registry import build_from_config
from .config import ModelConfig, normalize_share
from .random import spawn_agent_rngs, spawn_rngs
from .sim import draw_financing_series, draw_joint_returns, prepare_return_shocks
from .sim.covariance import build_cov_matrix
from .sim.metrics import summary_table
from .sim.params import build_financing_params, build_return_params, build_simulation_params
from .simulations import simulate_agents
from .types import GeneratorLike
from .validators import select_vol_regime_sigma


def progress_bar(iterable: Any, total: Optional[int] = None, desc: Optional[str] = None) -> Any:
    if _HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc)  # type: ignore[name-defined]
    return iterable


def generate_parameter_combinations(cfg: ModelConfig) -> Iterator[Dict[str, Any]]:
    """Generate parameter combinations based on ``analysis_mode``."""
    if cfg.analysis_mode == "capital":
        for ext_pct in np.arange(
            0,
            cfg.max_external_combined_pct + cfg.external_step_size_pct,
            cfg.external_step_size_pct,
        ):
            for act_pct in np.arange(
                0,
                ext_pct + cfg.external_step_size_pct,
                cfg.external_step_size_pct,
            ):
                ext_pa_pct = ext_pct - act_pct
                internal_pct = 100 - ext_pct
                yield {
                    "external_pa_capital": (ext_pa_pct / 100) * cfg.total_fund_capital,
                    "active_ext_capital": (act_pct / 100) * cfg.total_fund_capital,
                    "internal_pa_capital": (internal_pct / 100) * cfg.total_fund_capital,
                }
    elif cfg.analysis_mode == "returns":
        for mu_H in np.arange(
            cfg.in_house_return_min_pct,
            cfg.in_house_return_max_pct + cfg.in_house_return_step_pct,
            cfg.in_house_return_step_pct,
        ):
            for sigma_H in np.arange(
                cfg.in_house_vol_min_pct,
                cfg.in_house_vol_max_pct + cfg.in_house_vol_step_pct,
                cfg.in_house_vol_step_pct,
            ):
                for mu_E in np.arange(
                    cfg.alpha_ext_return_min_pct,
                    cfg.alpha_ext_return_max_pct + cfg.alpha_ext_return_step_pct,
                    cfg.alpha_ext_return_step_pct,
                ):
                    for sigma_E in np.arange(
                        cfg.alpha_ext_vol_min_pct,
                        cfg.alpha_ext_vol_max_pct + cfg.alpha_ext_vol_step_pct,
                        cfg.alpha_ext_vol_step_pct,
                    ):
                        yield {
                            "mu_H": mu_H / 100,
                            "sigma_H": sigma_H / 100,
                            "mu_E": mu_E / 100,
                            "sigma_E": sigma_E / 100,
                        }
    elif cfg.analysis_mode == "alpha_shares":
        for theta_extpa in np.arange(
            cfg.external_pa_alpha_min_pct,
            cfg.external_pa_alpha_max_pct + cfg.external_pa_alpha_step_pct,
            cfg.external_pa_alpha_step_pct,
        ):
            for active_share in np.arange(
                cfg.active_share_min_pct,
                cfg.active_share_max_pct + cfg.active_share_step_pct,
                cfg.active_share_step_pct,
            ):
                yield {
                    "theta_extpa": normalize_share(theta_extpa),
                    "active_share": normalize_share(active_share),
                }
    elif cfg.analysis_mode == "vol_mult":
        for sd_mult in np.arange(
            cfg.sd_multiple_min,
            cfg.sd_multiple_max + cfg.sd_multiple_step,
            cfg.sd_multiple_step,
        ):
            yield {
                "sigma_H": cfg.sigma_H * sd_mult,
                "sigma_E": cfg.sigma_E * sd_mult,
                "sigma_M": cfg.sigma_M * sd_mult,
            }
    else:
        raise ValueError(f"Unsupported analysis mode: {cfg.analysis_mode}")


logger = logging.getLogger(__name__)

"""Module-level cached empty DataFrame used for sweep results shape."""
EMPTY_RESULTS_COLUMNS: pd.Index = pd.Index(
    [
        "Agent",
        "AnnReturn",
        "AnnVol",
        "VaR",
        "CVaR",
        "MaxDD",
        "TimeUnderWater",
        "BreachProb",
        "BreachCount",
        "ShortfallProb",
        "TE",
        "combination_id",
    ],
    dtype="object",
)
_EMPTY_RESULTS_DF: pd.DataFrame = pd.DataFrame(columns=EMPTY_RESULTS_COLUMNS)


def _get_empty_results_dataframe() -> pd.DataFrame:
    """Return a copy of the cached empty DataFrame for sweep results."""
    return _EMPTY_RESULTS_DF.copy()


def run_parameter_sweep(
    cfg: ModelConfig,
    index_series: pd.Series,
    rng_returns: GeneratorLike,
    fin_rngs: Dict[str, GeneratorLike],
    seed: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Run the parameter sweep and collect results.

    Parameters
    ----------
    progress:
        Optional callback accepting ``(current, total)`` to report progress. When
        ``None``, a ``tqdm`` progress bar is displayed.
    """
    results: List[Dict[str, Any]] = []

    mu_idx = float(index_series.mean())
    idx_sigma, _, _ = select_vol_regime_sigma(
        index_series,
        regime=cfg.vol_regime,
        window=cfg.vol_regime_window,
    )
    n_samples = int(len(index_series))

    # Pre-compute combinations for progress tracking
    combos = list(generate_parameter_combinations(cfg))
    total = len(combos)
    logger.info("Starting parameter sweep", extra={"total_combinations": total})

    override_keys: set[str] = set()
    for overrides in combos:
        override_keys.update(overrides.keys())
    shock_incompatible_keys = {
        "rho_idx_H",
        "rho_idx_E",
        "rho_idx_M",
        "rho_H_E",
        "rho_H_M",
        "rho_E_M",
        "return_distribution",
        "return_t_df",
        "return_copula",
        "return_distribution_idx",
        "return_distribution_H",
        "return_distribution_E",
        "return_distribution_M",
    }
    financing_keys = {
        "internal_financing_mean_month",
        "internal_financing_sigma_month",
        "internal_spike_prob",
        "internal_spike_factor",
        "ext_pa_financing_mean_month",
        "ext_pa_financing_sigma_month",
        "ext_pa_spike_prob",
        "ext_pa_spike_factor",
        "act_ext_financing_mean_month",
        "act_ext_financing_sigma_month",
        "act_ext_spike_prob",
        "act_ext_spike_factor",
    }
    reuse_return_shocks = not override_keys.intersection(shock_incompatible_keys)
    reuse_financing_series = not override_keys.intersection(financing_keys)

    # Common random numbers: reset RNGs before each combination so parameter
    # changes are compared against identical random draws. When a master seed
    # is provided, derive the baseline RNG state from that seed.
    if seed is None:
        rng_returns_state = copy.deepcopy(rng_returns.bit_generator.state)
        fin_rng_states = {
            name: copy.deepcopy(rng.bit_generator.state) for name, rng in fin_rngs.items()
        }
    else:
        base_rng_returns = spawn_rngs(seed, 1)[0]
        base_fin_rngs = spawn_agent_rngs(seed, list(fin_rngs.keys()))
        rng_returns_state = copy.deepcopy(base_rng_returns.bit_generator.state)
        fin_rng_states = {
            name: copy.deepcopy(base_fin_rngs[name].bit_generator.state) for name in fin_rngs.keys()
        }

    return_shocks = None
    if reuse_return_shocks:
        shock_params = build_return_params(cfg, mu_idx=mu_idx, idx_sigma=idx_sigma)
        rng_returns_base = spawn_rngs(None, 1)[0]
        rng_returns_base.bit_generator.state = copy.deepcopy(rng_returns_state)
        return_shocks = prepare_return_shocks(
            n_months=cfg.N_MONTHS,
            n_sim=cfg.N_SIMULATIONS,
            params=shock_params,
            rng=rng_returns_base,
        )

    financing_series = None
    if reuse_financing_series:
        financing_params = build_financing_params(cfg)
        fin_rngs_base: Dict[str, GeneratorLike] = {}
        for name in fin_rngs.keys():
            tmp_rng = spawn_rngs(None, 1)[0]
            tmp_rng.bit_generator.state = copy.deepcopy(fin_rng_states[name])
            fin_rngs_base[name] = tmp_rng
        financing_series = draw_financing_series(
            n_months=cfg.N_MONTHS,
            n_sim=cfg.N_SIMULATIONS,
            params=financing_params,
            rngs=fin_rngs_base,
        )

    iterator = enumerate(combos)
    if progress is None:
        iterator = enumerate(progress_bar(combos, total=total, desc="sweep"))

    for i, overrides in iterator:
        if return_shocks is None:
            rng_returns.bit_generator.state = copy.deepcopy(rng_returns_state)
        if financing_series is None:
            for name, rng in fin_rngs.items():
                rng.bit_generator.state = copy.deepcopy(fin_rng_states[name])

        mod_cfg = cfg.model_copy(update=overrides)

        build_cov_matrix(
            mod_cfg.rho_idx_H,
            mod_cfg.rho_idx_E,
            mod_cfg.rho_idx_M,
            mod_cfg.rho_H_E,
            mod_cfg.rho_H_M,
            mod_cfg.rho_E_M,
            idx_sigma,
            mod_cfg.sigma_H,
            mod_cfg.sigma_E,
            mod_cfg.sigma_M,
            covariance_shrinkage=mod_cfg.covariance_shrinkage,
            n_samples=n_samples,
        )
        params = build_simulation_params(mod_cfg, mu_idx=mu_idx, idx_sigma=idx_sigma)

        r_beta, r_H, r_E, r_M = draw_joint_returns(
            n_months=mod_cfg.N_MONTHS,
            n_sim=mod_cfg.N_SIMULATIONS,
            params=params,
            rng=None if return_shocks is not None else rng_returns,
            shocks=return_shocks,
        )
        if financing_series is None:
            f_int, f_ext, f_act = draw_financing_series(
                n_months=mod_cfg.N_MONTHS,
                n_sim=mod_cfg.N_SIMULATIONS,
                params=params,
                rngs=fin_rngs,
            )
        else:
            f_int, f_ext, f_act = financing_series

        agents = build_from_config(mod_cfg)
        returns = simulate_agents(
            agents,
            r_beta,
            r_H,
            r_E,
            r_M,
            f_int,
            f_ext,
            f_act,
        )

        summary = summary_table(returns, benchmark="Base")
        results.append(
            {
                "combination_id": i,
                "parameters": overrides,
                "summary": summary,
            }
        )

        if progress is not None:
            progress(i + 1, total)
        else:
            logger.debug("sweep step", extra={"current": i + 1, "total": total})

    logger.info("Parameter sweep complete")
    return results


# ---------------------------------------------------------------------------
# Cached sweep and result helpers

_SWEEP_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _make_cache_key(cfg: ModelConfig, index_series: pd.Series, seed: int) -> str:
    """Return a hash key for caching parameter sweeps."""
    cfg_json = json.dumps(cfg.model_dump(), sort_keys=True)
    # Use getattr to avoid static checker complaining about pandas.util access
    hash_fn = getattr(pd, "util").hash_pandas_object  # type: ignore[attr-defined]
    idx_hash = hashlib.sha256(hash_fn(index_series).values.tobytes()).hexdigest()
    return hashlib.sha256((cfg_json + idx_hash + str(seed)).encode()).hexdigest()


def run_parameter_sweep_cached(
    cfg: ModelConfig,
    index_series: pd.Series,
    seed: int,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Run ``run_parameter_sweep`` with simple in-memory caching.

    The cache key is derived from the configuration, index series and seed.
    Subsequent calls with identical parameters return the cached results
    without re-running the simulation.
    """
    key = _make_cache_key(cfg, index_series, seed)
    if key not in _SWEEP_CACHE:
        rng_returns = spawn_rngs(seed, 1)[0]
        fin_rngs = spawn_agent_rngs(seed, ["internal", "external_pa", "active_ext"])
        _SWEEP_CACHE[key] = run_parameter_sweep(
            cfg, index_series, rng_returns, fin_rngs, seed=seed, progress=progress
        )
    return _SWEEP_CACHE[key]


def sweep_results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten sweep results into a single DataFrame.

    Each combination's summary metrics are combined with its parameters and
    identifier to form one row per agent and parameter combination.
    """
    frames: List[pd.DataFrame] = []
    for res in results:
        summary = res["summary"].copy()
        for key, val in res["parameters"].items():
            summary[key] = val
        summary["combination_id"] = res["combination_id"]
        frames.append(summary)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return _get_empty_results_dataframe()


__all__ = [
    "generate_parameter_combinations",
    "run_parameter_sweep",
    "run_parameter_sweep_cached",
    "sweep_results_to_dataframe",
]
