from __future__ import annotations

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
from .config import ModelConfig
from .random import spawn_agent_rngs, spawn_rngs
from .sim import draw_financing_series, draw_joint_returns
from .sim.covariance import build_cov_matrix
from .sim.metrics import summary_table
from .simulations import simulate_agents


def progress_bar(
    iterable: Any, total: Optional[int] = None, desc: Optional[str] = None
) -> Any:
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
                    "internal_pa_capital": (internal_pct / 100)
                    * cfg.total_fund_capital,
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
                    "theta_extpa": theta_extpa / 100,
                    "active_share": active_share,
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

# Create a cached empty DataFrame with expected columns to avoid repeated creation
# This provides a significant performance improvement for empty result cases
_EMPTY_RESULTS_DF: pd.DataFrame | None = None


def _get_empty_results_dataframe() -> pd.DataFrame:
    """Return a cached empty DataFrame with expected columns for sweep results."""
    global _EMPTY_RESULTS_DF
    if _EMPTY_RESULTS_DF is None:
        # Define the expected columns from summary_table plus parameters and combination_id
        # This matches the structure returned by summary_table in pa_core.sim.metrics
        columns = [
            "Agent", "AnnReturn", "AnnVol", "VaR", "CVaR", "MaxDD", 
            "TimeUnderWater", "BreachProb", "BreachCount", "ShortfallProb", "TE",
            "combination_id"
        ]
        _EMPTY_RESULTS_DF = pd.DataFrame(columns=columns)
    return _EMPTY_RESULTS_DF.copy()  # Return a copy to avoid mutations


def run_parameter_sweep(
    cfg: ModelConfig,
    index_series: pd.Series,
    rng_returns: np.random.Generator,
    fin_rngs: Dict[str, np.random.Generator],
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
    idx_sigma = float(index_series.std(ddof=1))

    # Pre-compute combinations for progress tracking
    combos = list(generate_parameter_combinations(cfg))
    total = len(combos)
    logger.info("Starting parameter sweep", extra={"total_combinations": total})

    iterator = enumerate(combos)
    if progress is None:
        iterator = enumerate(progress_bar(combos, total=total, desc="sweep"))

    for i, overrides in iterator:
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
        )
        params = {
            "mu_idx_month": mu_idx / 12,
            "default_mu_H": mod_cfg.mu_H / 12,
            "default_mu_E": mod_cfg.mu_E / 12,
            "default_mu_M": mod_cfg.mu_M / 12,
            "idx_sigma_month": idx_sigma / 12,
            "default_sigma_H": mod_cfg.sigma_H / 12,
            "default_sigma_E": mod_cfg.sigma_E / 12,
            "default_sigma_M": mod_cfg.sigma_M / 12,
            "rho_idx_H": mod_cfg.rho_idx_H,
            "rho_idx_E": mod_cfg.rho_idx_E,
            "rho_idx_M": mod_cfg.rho_idx_M,
            "rho_H_E": mod_cfg.rho_H_E,
            "rho_H_M": mod_cfg.rho_H_M,
            "rho_E_M": mod_cfg.rho_E_M,
            "internal_financing_mean_month": mod_cfg.internal_financing_mean_month,
            "internal_financing_sigma_month": mod_cfg.internal_financing_sigma_month,
            "internal_spike_prob": mod_cfg.internal_spike_prob,
            "internal_spike_factor": mod_cfg.internal_spike_factor,
            "ext_pa_financing_mean_month": mod_cfg.ext_pa_financing_mean_month,
            "ext_pa_financing_sigma_month": mod_cfg.ext_pa_financing_sigma_month,
            "ext_pa_spike_prob": mod_cfg.ext_pa_spike_prob,
            "ext_pa_spike_factor": mod_cfg.ext_pa_spike_factor,
            "act_ext_financing_mean_month": mod_cfg.act_ext_financing_mean_month,
            "act_ext_financing_sigma_month": mod_cfg.act_ext_financing_sigma_month,
            "act_ext_spike_prob": mod_cfg.act_ext_spike_prob,
            "act_ext_spike_factor": mod_cfg.act_ext_spike_factor,
        }

        r_beta, r_H, r_E, r_M = draw_joint_returns(
            n_months=mod_cfg.N_MONTHS,
            n_sim=mod_cfg.N_SIMULATIONS,
            params=params,
            rng=rng_returns,
        )
        f_int, f_ext, f_act = draw_financing_series(
            n_months=mod_cfg.N_MONTHS,
            n_sim=mod_cfg.N_SIMULATIONS,
            params=params,
            rngs=fin_rngs,
        )

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
            cfg, index_series, rng_returns, fin_rngs, progress=progress
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
