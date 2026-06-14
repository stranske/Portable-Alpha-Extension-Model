from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

# tqdm is optional; provide a no-op fallback wrapper to avoid hard dependency at import time
try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    _HAS_TQDM = False

from .agents.registry import build_from_config
from .config import ModelConfig, SweepConfig, SweepParameter, normalize_share, validate_financing_spikes
from .contracts import SUMMARY_NUMERIC_COLUMNS
from .random import spawn_agent_rngs, spawn_rngs
from .sim import (
    draw_financing_series,
    draw_joint_returns,
    prepare_return_shocks,
    resolve_internal_pa_financing_series,
)
from .sim.covariance import build_cov_matrix
from .sim.metrics import summary_table
from .sim.params import (
    build_covariance_return_overrides,
    build_financing_params,
    build_params,
    resolve_covariance_inputs,
)
from .sim.regimes import build_regime_draw_params, resolve_regime_start, simulate_regime_paths
from .simulations import simulate_agents
from .types import GeneratorLike, SweepResult
from .units import (
    convert_mean,
    convert_volatility,
    get_index_series_unit,
    normalize_index_series,
    normalize_return_inputs,
)
from .validators import select_vol_regime_sigma


def progress_bar(iterable: Any, total: Optional[int] = None, desc: Optional[str] = None) -> Any:
    if _HAS_TQDM:
        return _tqdm(iterable, total=total, desc=desc)
    return iterable


def _count_range(start: float, stop: float, step: float) -> int:
    values = np.arange(start, stop + step, step)
    return int(len(values))


def _estimate_total_combinations(cfg: ModelConfig) -> int:
    if cfg.sweep is not None:
        if cfg.sweep.method == "random":
            return int(cfg.sweep.samples or 0)
        total = 1
        for param in cfg.sweep.parameters.values():
            if param.values is not None:
                count = len(param.values)
            else:
                if param.min is None or param.max is None or param.step is None:
                    raise ValueError(
                        "Sweep parameter requires 'min', 'max', and 'step' when 'values' is not set"
                    )
                count = _count_range(float(param.min), float(param.max), float(param.step))
            total *= count
        return int(total)

    if cfg.analysis_mode == "capital":
        n = _count_range(
            0.0, float(cfg.max_external_combined_pct), float(cfg.external_step_size_pct)
        )
        return int(n * (n + 1) / 2)
    if cfg.analysis_mode == "returns":
        return int(
            _count_range(
                float(cfg.in_house_return_min_pct),
                float(cfg.in_house_return_max_pct),
                float(cfg.in_house_return_step_pct),
            )
            * _count_range(
                float(cfg.in_house_vol_min_pct),
                float(cfg.in_house_vol_max_pct),
                float(cfg.in_house_vol_step_pct),
            )
            * _count_range(
                float(cfg.alpha_ext_return_min_pct),
                float(cfg.alpha_ext_return_max_pct),
                float(cfg.alpha_ext_return_step_pct),
            )
            * _count_range(
                float(cfg.alpha_ext_vol_min_pct),
                float(cfg.alpha_ext_vol_max_pct),
                float(cfg.alpha_ext_vol_step_pct),
            )
        )
    if cfg.analysis_mode == "alpha_shares":
        return int(
            _count_range(
                float(cfg.external_pa_alpha_min_pct),
                float(cfg.external_pa_alpha_max_pct),
                float(cfg.external_pa_alpha_step_pct),
            )
            * _count_range(
                float(cfg.active_share_min_pct),
                float(cfg.active_share_max_pct),
                float(cfg.active_share_step_pct),
            )
        )
    if cfg.analysis_mode == "vol_mult":
        return _count_range(
            float(cfg.sd_multiple_min), float(cfg.sd_multiple_max), float(cfg.sd_multiple_step)
        )
    raise ValueError(f"Unsupported analysis mode: {cfg.analysis_mode}")


def _override_keys_for_config(cfg: ModelConfig) -> set[str]:
    if cfg.sweep is not None:
        return set(cfg.sweep.parameters.keys())
    if cfg.analysis_mode == "capital":
        return {"external_pa_capital", "active_ext_capital", "internal_pa_capital"}
    if cfg.analysis_mode == "returns":
        return {"mu_H", "sigma_H", "mu_E", "sigma_E"}
    if cfg.analysis_mode == "alpha_shares":
        return {"theta_extpa", "active_share"}
    if cfg.analysis_mode == "vol_mult":
        return {"sigma_H", "sigma_E", "sigma_M"}
    return set()


def _iter_sweep_grid(sweep: SweepConfig) -> Iterator[Dict[str, Any]]:
    names: List[str] = []
    values: List[List[float]] = []
    for name, param in sweep.parameters.items():
        if param.values is not None:
            param_values = list(param.values)
        else:
            if param.min is None or param.max is None or param.step is None:
                raise ValueError(
                    "Sweep parameter requires 'min', 'max', and 'step' when 'values' is not set"
                )
            param_values = list(np.arange(param.min, param.max + param.step, param.step))
        names.append(name)
        values.append(param_values)

    for combo in product(*values):
        yield dict(zip(names, combo))


def _sample_sweep_value(param: SweepParameter, rng: np.random.Generator) -> float:
    if param.values is not None:
        return float(rng.choice(param.values))
    if param.step is not None:
        if param.min is None or param.max is None:
            raise ValueError("Sweep parameter requires 'min' and 'max' for stepped sampling")
        grid = np.arange(param.min, param.max + param.step, param.step)
        return float(rng.choice(grid))
    if param.min is None or param.max is None:
        raise ValueError("Sweep parameter requires 'min' and 'max' for uniform sampling")
    return float(rng.uniform(param.min, param.max))


def _iter_sweep_random(sweep: SweepConfig) -> Iterator[Dict[str, Any]]:
    rng = np.random.default_rng(sweep.seed)
    for _ in range(sweep.samples or 0):
        combo: Dict[str, Any] = {}
        for name, param in sweep.parameters.items():
            combo[name] = _sample_sweep_value(param, rng)
        yield combo


def generate_parameter_combinations(cfg: ModelConfig) -> Iterator[Dict[str, Any]]:
    """Generate parameter combinations based on ``analysis_mode``."""
    if cfg.sweep is not None:
        if cfg.sweep.method == "grid":
            yield from _iter_sweep_grid(cfg.sweep)
            return
        yield from _iter_sweep_random(cfg.sweep)
        return
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
                        mu_H_val = convert_mean(mu_H / 100.0, from_unit="annual", to_unit="monthly")
                        sigma_H_val = convert_volatility(
                            sigma_H / 100.0, from_unit="annual", to_unit="monthly"
                        )
                        mu_E_val = convert_mean(mu_E / 100.0, from_unit="annual", to_unit="monthly")
                        sigma_E_val = convert_volatility(
                            sigma_E / 100.0, from_unit="annual", to_unit="monthly"
                        )
                        yield {
                            "mu_H": mu_H_val,
                            "sigma_H": sigma_H_val,
                            "mu_E": mu_E_val,
                            "sigma_E": sigma_E_val,
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


@dataclass(slots=True)
class SweepRunner:
    """Execute parameter sweeps with a lightweight wrapper around sweep helpers."""

    config: ModelConfig
    index_series: pd.Series | pd.DataFrame
    seed: Optional[int] = None
    legacy_agent_rng: bool = False
    progress: Optional[Callable[[int, int], None]] = None
    substream_ids: Mapping[str, str] | None = None

    def iter_combinations(self) -> Iterator[Dict[str, Any]]:
        return generate_parameter_combinations(self.config)

    def run(self) -> List[SweepResult]:
        from .sim.simulation_initialization import initialize_sweep_rngs

        idx_series = self.index_series
        if isinstance(idx_series, pd.DataFrame):
            idx_series = idx_series.squeeze()
        if not isinstance(idx_series, pd.Series):
            try:
                idx_series = pd.Series(idx_series)
            except Exception as exc:
                raise ValueError("Index data must be convertible to pandas Series") from exc

        rng_bundle = initialize_sweep_rngs(
            self.seed,
            legacy_agent_rng=self.legacy_agent_rng,
        )
        self.substream_ids = rng_bundle.substream_ids
        return run_parameter_sweep(
            self.config,
            idx_series,
            rng_bundle.rng_returns,
            rng_bundle.rngs_financing,
            seed=rng_bundle.seed,
            rng_regime=rng_bundle.rng_regime,
            progress=self.progress,
        )


"""Module-level cached empty DataFrame used for sweep results shape."""
EMPTY_RESULTS_COLUMNS: pd.Index = pd.Index(
    [
        "Agent",
        "terminal_AnnReturn",
        "monthly_AnnVol",
        "monthly_VaR",
        "monthly_CVaR",
        "terminal_CVaR",
        "monthly_MaxDD",
        "monthly_TimeUnderWater",
        "monthly_BreachProb",
        "monthly_BreachCountPath0",
        "terminal_ShortfallProb",
        "monthly_TE",
        "combination_id",
    ],
    dtype="object",
)
_EMPTY_RESULTS_DF: pd.DataFrame = pd.DataFrame(columns=EMPTY_RESULTS_COLUMNS)


def _get_empty_results_dataframe() -> pd.DataFrame:
    """Return a copy of the cached empty DataFrame for sweep results."""
    return _EMPTY_RESULTS_DF.copy()


def _derive_regime_rng(rng_returns: GeneratorLike) -> GeneratorLike:
    """Derive a regime RNG deterministically from ``rng_returns``'s state.

    Used when neither a master ``seed`` nor an explicit ``rng_regime`` is
    supplied: the regime path generator is seeded from a stable hash of the
    ``rng_returns`` bit-generator state so repeated runs with the same seeded
    ``rng_returns`` reproduce identical regime paths, while remaining an
    independent stream from the return draws.
    """
    state_repr = json.dumps(rng_returns.bit_generator.state, sort_keys=True, default=str)
    digest = hashlib.sha256(("pa_core.regime:" + state_repr).encode("utf-8")).digest()
    entropy = int.from_bytes(digest, "big")
    return spawn_rngs(entropy, 1)[0]


def run_parameter_sweep(
    cfg: ModelConfig,
    index_series: pd.Series,
    rng_returns: GeneratorLike,
    fin_rngs: Mapping[str, GeneratorLike],
    seed: Optional[int] = None,
    rng_regime: GeneratorLike | None = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[SweepResult]:
    """Run the parameter sweep and collect results.

    Index returns are normalised to monthly units before simulation, and
    summary metrics (terminal_AnnReturn/monthly_AnnVol/monthly_TE) are annualised
    from monthly returns.

    Parameters
    ----------
    progress:
        Optional callback accepting ``(current, total)`` to report progress. When
        ``None``, a ``tqdm`` progress bar is displayed.
    """
    results: List[SweepResult] = []
    if rng_regime is None:
        if seed is not None:
            rng_regime = spawn_rngs(seed, 2)[1]
        else:
            # No master seed and no explicit regime RNG: derive one
            # deterministically from the supplied ``rng_returns`` so that direct
            # callers that seed ``rng_returns``/``fin_rngs`` (the established
            # reproducibility pattern) still get repeatable regime paths instead
            # of OS entropy. Reading ``bit_generator.state`` does not advance
            # ``rng_returns``, so this leaves the downstream draws untouched.
            rng_regime = _derive_regime_rng(rng_returns)

    index_series = normalize_index_series(pd.Series(index_series), get_index_series_unit())
    mu_idx = float(index_series.mean())
    idx_sigma, _, _ = select_vol_regime_sigma(
        index_series,
        regime=cfg.vol_regime,
        window=cfg.vol_regime_window,
    )
    n_samples = int(len(index_series))

    total = _estimate_total_combinations(cfg)
    logger.info("Starting parameter sweep", extra={"total_combinations": total})

    override_keys = _override_keys_for_config(cfg)
    shock_incompatible_keys = {
        "rho_idx_H",
        "rho_idx_E",
        "rho_idx_M",
        "rho_H_E",
        "rho_H_M",
        "rho_E_M",
        "correlation_repair_mode",
        "correlation_repair_shrinkage",
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
        "financing_mode",
    }
    sigma_override_keys = {"sigma_H", "sigma_E", "sigma_M"}
    return_param_keys = {
        "return_unit",
        "mu_H",
        "mu_E",
        "mu_M",
        "sigma_H",
        "sigma_E",
        "sigma_M",
        "rho_idx_H",
        "rho_idx_E",
        "rho_idx_M",
        "rho_H_E",
        "rho_H_M",
        "rho_E_M",
        "correlation_repair_mode",
        "correlation_repair_shrinkage",
        "return_distribution",
        "return_t_df",
        "return_copula",
        "return_distribution_idx",
        "return_distribution_H",
        "return_distribution_E",
        "return_distribution_M",
        "covariance_shrinkage",
    }
    reuse_return_shocks = not override_keys.intersection(shock_incompatible_keys)
    reuse_financing_series = not override_keys.intersection(financing_keys)
    if cfg.regimes is not None:
        reuse_return_shocks = False
    if cfg.covariance_shrinkage != "none" and override_keys.intersection(sigma_override_keys):
        reuse_return_shocks = False
    returns_static = not override_keys.intersection(return_param_keys)
    financing_static = not override_keys.intersection(financing_keys)

    # Common random numbers: reset RNGs before each combination so parameter
    # changes are compared against identical random draws. When a master seed
    # is provided, derive the baseline RNG state from that seed.
    if seed is None:
        rng_returns_state = copy.deepcopy(rng_returns.bit_generator.state)
        rng_regime_state = copy.deepcopy(rng_regime.bit_generator.state)
        fin_rng_states = {
            name: copy.deepcopy(rng.bit_generator.state) for name, rng in fin_rngs.items()
        }
    else:
        base_rng_returns, base_rng_regime = spawn_rngs(seed, 2)
        base_fin_rngs = spawn_agent_rngs(seed, list(fin_rngs.keys()))
        rng_returns_state = copy.deepcopy(base_rng_returns.bit_generator.state)
        rng_regime_state = copy.deepcopy(base_rng_regime.bit_generator.state)
        fin_rng_states = {
            name: copy.deepcopy(base_fin_rngs[name].bit_generator.state) for name in fin_rngs.keys()
        }

    base_sigma = None
    base_corr = None
    base_params = None
    base_regime_params = None
    base_regime_paths = None
    if returns_static or reuse_return_shocks:
        base_return_inputs = normalize_return_inputs(cfg)
        sigma_h = float(base_return_inputs["sigma_H"])
        sigma_e = float(base_return_inputs["sigma_E"])
        sigma_m = float(base_return_inputs["sigma_M"])
        base_cov = build_cov_matrix(
            cfg.rho_idx_H,
            cfg.rho_idx_E,
            cfg.rho_idx_M,
            cfg.rho_H_E,
            cfg.rho_H_M,
            cfg.rho_E_M,
            idx_sigma,
            sigma_h,
            sigma_e,
            sigma_m,
            covariance_shrinkage=cfg.covariance_shrinkage,
            n_samples=n_samples,
        )
        base_sigma, base_corr = resolve_covariance_inputs(
            base_cov,
            idx_sigma=idx_sigma,
            sigma_h=sigma_h,
            sigma_e=sigma_e,
            sigma_m=sigma_m,
            rho_idx_H=cfg.rho_idx_H,
            rho_idx_E=cfg.rho_idx_E,
            rho_idx_M=cfg.rho_idx_M,
            rho_H_E=cfg.rho_H_E,
            rho_H_M=cfg.rho_H_M,
            rho_E_M=cfg.rho_E_M,
        )
        base_params = build_params(
            cfg,
            mu_idx=mu_idx,
            idx_sigma=float(base_sigma[0]),
            return_overrides=build_covariance_return_overrides(base_sigma, base_corr),
        )
        if cfg.regimes is not None:
            if cfg.regime_transition is None:
                raise ValueError("regime_transition is required when regimes are specified")
            base_regime_params, _labels = build_regime_draw_params(
                cfg,
                mu_idx=mu_idx,
                idx_sigma=float(base_sigma[0]),
                n_samples=n_samples,
            )
            rng_regime_base = spawn_rngs(None, 1)[0]
            rng_regime_base.bit_generator.state = copy.deepcopy(rng_regime_state)
            base_regime_paths = simulate_regime_paths(
                n_sim=cfg.N_SIMULATIONS,
                n_months=cfg.N_MONTHS,
                transition=cfg.regime_transition,
                start_state=resolve_regime_start(cfg),
                rng=rng_regime_base,
            )

    return_shocks = None
    if reuse_return_shocks:
        if base_params is None:
            raise RuntimeError("Base parameters are required to prepare return shocks")
        rng_returns_base = spawn_rngs(None, 1)[0]
        rng_returns_base.bit_generator.state = copy.deepcopy(rng_returns_state)
        return_shocks = prepare_return_shocks(
            n_months=cfg.N_MONTHS,
            n_sim=cfg.N_SIMULATIONS,
            params=base_params,
            rng=rng_returns_base,
        )

    financing_series = None
    if reuse_financing_series:
        idx_sigma_fin = float(base_sigma[0]) if base_sigma is not None else float(idx_sigma)
        financing_params = build_params(cfg, mu_idx=mu_idx, idx_sigma=idx_sigma_fin)
        fin_rngs_base: Dict[str, GeneratorLike] = {}
        for name in fin_rngs.keys():
            tmp_rng = spawn_rngs(None, 1)[0]
            tmp_rng.bit_generator.state = copy.deepcopy(fin_rng_states[name])
            fin_rngs_base[name] = tmp_rng
        financing_series = draw_financing_series(
            n_months=cfg.N_MONTHS,
            n_sim=cfg.N_SIMULATIONS,
            params=financing_params,
            financing_mode=cfg.financing_mode,
            rngs=fin_rngs_base,
        )

    combos_iter: Any = generate_parameter_combinations(cfg)
    if progress is None:
        combos_iter = progress_bar(combos_iter, total=total, desc="sweep")
    iterator = enumerate(combos_iter)

    for i, overrides in iterator:
        if return_shocks is None:
            rng_returns.bit_generator.state = copy.deepcopy(rng_returns_state)
            rng_regime.bit_generator.state = copy.deepcopy(rng_regime_state)
        if financing_series is None:
            for name, rng in fin_rngs.items():
                rng.bit_generator.state = copy.deepcopy(fin_rng_states[name])

        # Build each sweep case with ``model_copy`` (no full re-validation) so the
        # grid can explore over-margin capital combinations as before, but re-run
        # the inert-spike check so a sweep override cannot silently introduce a
        # spike with zero financing volatility.
        mod_cfg = cfg.model_copy(update=overrides)
        validate_financing_spikes(mod_cfg)

        if returns_static:
            if base_params is None:
                raise RuntimeError("Base parameters are required for static return sweeps")
            if financing_static:
                params = base_params
            else:
                params = dict(base_params)
                params.update(build_financing_params(mod_cfg))
        else:
            return_inputs = normalize_return_inputs(mod_cfg)
            sigma_h = float(return_inputs["sigma_H"])
            sigma_e = float(return_inputs["sigma_E"])
            sigma_m = float(return_inputs["sigma_M"])

            cov = build_cov_matrix(
                mod_cfg.rho_idx_H,
                mod_cfg.rho_idx_E,
                mod_cfg.rho_idx_M,
                mod_cfg.rho_H_E,
                mod_cfg.rho_H_M,
                mod_cfg.rho_E_M,
                idx_sigma,
                sigma_h,
                sigma_e,
                sigma_m,
                covariance_shrinkage=mod_cfg.covariance_shrinkage,
                n_samples=n_samples,
            )
            sigma_vec, corr_mat = resolve_covariance_inputs(
                cov,
                idx_sigma=idx_sigma,
                sigma_h=sigma_h,
                sigma_e=sigma_e,
                sigma_m=sigma_m,
                rho_idx_H=mod_cfg.rho_idx_H,
                rho_idx_E=mod_cfg.rho_idx_E,
                rho_idx_M=mod_cfg.rho_idx_M,
                rho_H_E=mod_cfg.rho_H_E,
                rho_H_M=mod_cfg.rho_H_M,
                rho_E_M=mod_cfg.rho_E_M,
            )
            params = build_params(
                mod_cfg,
                mu_idx=mu_idx,
                idx_sigma=float(sigma_vec[0]),
                return_overrides=build_covariance_return_overrides(sigma_vec, corr_mat),
            )

        regime_params = None
        regime_paths = None
        if mod_cfg.regimes is not None:
            if mod_cfg.regime_transition is None:
                raise ValueError("regime_transition is required when regimes are specified")
            if returns_static and base_regime_params is not None and base_regime_paths is not None:
                regime_params = base_regime_params
                regime_paths = base_regime_paths
            else:
                regime_params = build_regime_draw_params(
                    mod_cfg,
                    mu_idx=mu_idx,
                    idx_sigma=float(params["idx_sigma_month"]),
                    n_samples=n_samples,
                )[0]
                regime_paths = simulate_regime_paths(
                    n_sim=mod_cfg.N_SIMULATIONS,
                    n_months=mod_cfg.N_MONTHS,
                    transition=mod_cfg.regime_transition,
                    start_state=resolve_regime_start(mod_cfg),
                    rng=rng_regime,
                )

        r_beta, r_H, r_E, r_M = draw_joint_returns(
            n_months=mod_cfg.N_MONTHS,
            n_sim=mod_cfg.N_SIMULATIONS,
            params=params,
            rng=None if return_shocks is not None else rng_returns,
            shocks=return_shocks,
            regime_paths=regime_paths,
            regime_params=regime_params,
        )
        if financing_series is None:
            f_int, f_ext, f_act = draw_financing_series(
                n_months=mod_cfg.N_MONTHS,
                n_sim=mod_cfg.N_SIMULATIONS,
                params=params,
                financing_mode=mod_cfg.financing_mode,
                rngs=fin_rngs,
            )
        else:
            f_int, f_ext, f_act = financing_series
        f_internal_pa = resolve_internal_pa_financing_series(
            n_months=mod_cfg.N_MONTHS,
            n_sim=mod_cfg.N_SIMULATIONS,
            mean_month=getattr(mod_cfg, "internal_pa_financing_mean_month", 0.0),
            sigma_month=getattr(mod_cfg, "internal_pa_financing_sigma_month", 0.0),
            series=getattr(mod_cfg, "internal_pa_financing_series", None),
            index=getattr(mod_cfg, "internal_pa_financing_index", None),
            financing_mode=mod_cfg.financing_mode,
            rng=fin_rngs.get("internal") if fin_rngs else None,
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
            f_internal_pa,
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

SWEEP_CACHE_MAX_ENTRIES = 8
_SWEEP_CACHE: "OrderedDict[str, List[SweepResult]]" = OrderedDict()


def _enforce_sweep_cache_limit() -> None:
    if SWEEP_CACHE_MAX_ENTRIES < 1:
        return
    while len(_SWEEP_CACHE) > SWEEP_CACHE_MAX_ENTRIES:
        _SWEEP_CACHE.popitem(last=False)


def _make_cache_key(cfg: ModelConfig, index_series: pd.Series, seed: int) -> str:
    """Return a hash key for caching parameter sweeps."""
    cfg_json = json.dumps(cfg.model_dump(), sort_keys=True)
    # Use getattr to avoid static checker complaining about pandas.util access
    hash_fn = getattr(pd, "util").hash_pandas_object
    idx_hash = hashlib.sha256(hash_fn(index_series).values.tobytes()).hexdigest()
    return hashlib.sha256((cfg_json + idx_hash + str(seed)).encode()).hexdigest()


def run_parameter_sweep_cached(
    cfg: ModelConfig,
    index_series: pd.Series,
    seed: int,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[SweepResult]:
    """Run ``run_parameter_sweep`` with simple in-memory caching.

    The cache key is derived from the configuration, index series and seed.
    Subsequent calls with identical parameters return the cached results
    without re-running the simulation.
    """
    if SWEEP_CACHE_MAX_ENTRIES < 1:
        if _SWEEP_CACHE:
            _SWEEP_CACHE.clear()
        rng_returns = spawn_rngs(seed, 1)[0]
        fin_rngs = spawn_agent_rngs(seed, ["internal", "external_pa", "active_ext"])
        results = run_parameter_sweep(
            cfg, index_series, rng_returns, fin_rngs, seed=seed, progress=progress
        )
        if progress is not None:
            progress(len(results), len(results))
        return results
    key = _make_cache_key(cfg, index_series, seed)
    if key in _SWEEP_CACHE:
        _SWEEP_CACHE.move_to_end(key)
        _enforce_sweep_cache_limit()
    else:
        rng_returns = spawn_rngs(seed, 1)[0]
        fin_rngs = spawn_agent_rngs(seed, ["internal", "external_pa", "active_ext"])
        _SWEEP_CACHE[key] = run_parameter_sweep(
            cfg, index_series, rng_returns, fin_rngs, seed=seed, progress=progress
        )
        _enforce_sweep_cache_limit()
    results = _SWEEP_CACHE[key]
    if progress is not None:
        progress(len(results), len(results))
    return results


def clear_sweep_cache() -> None:
    """Clear cached parameter sweep results."""
    _SWEEP_CACHE.clear()


def sweep_results_to_dataframe(results: List[SweepResult]) -> pd.DataFrame:
    """Flatten sweep results into a single DataFrame.

    Each combination's summary metrics are combined with its parameters and
    identifier to form one row per agent and parameter combination.
    """
    frames: List[pd.DataFrame] = []
    for res in results:
        summary = res["summary"].copy()
        parameters = res.get("parameters", {})
        for key, val in parameters.items():
            summary[key] = val
        summary["combination_id"] = res["combination_id"]
        frames.append(summary)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return _get_empty_results_dataframe()


def aggregate_sweep_results(
    results: Sequence[SweepResult],
    *,
    percentiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Aggregate sweep summary metrics across combinations."""
    df = sweep_results_to_dataframe(list(results))
    percentiles = tuple(percentiles)
    if any(p < 0 or p > 1 for p in percentiles):
        raise ValueError("percentiles must be between 0 and 1")
    columns = ["Agent", "Metric", "Mean", "Std"]
    columns += [f"P{int(round(p * 100))}" for p in percentiles]
    if df.empty or "Agent" not in df.columns:
        return pd.DataFrame(columns=columns)

    metrics = [col for col in SUMMARY_NUMERIC_COLUMNS if col in df.columns]
    if not metrics:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, float | str]] = []
    grouped = df.groupby("Agent", dropna=False)
    for agent, group in grouped:
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            entry: dict[str, float | str] = {
                "Agent": str(agent),
                "Metric": metric,
                "Mean": float(values.mean()),
                "Std": float(values.std(ddof=1)),
            }
            for pct in percentiles:
                entry[f"P{int(round(pct * 100))}"] = float(values.quantile(pct))
            rows.append(entry)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


__all__ = [
    "generate_parameter_combinations",
    "run_parameter_sweep",
    "run_parameter_sweep_cached",
    "clear_sweep_cache",
    "sweep_results_to_dataframe",
    "aggregate_sweep_results",
    "SweepRunner",
]
