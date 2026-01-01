from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import Iterable, Literal, Protocol, Sequence, cast

import numpy as np
import pandas as pd

from .agents.registry import build_from_config
from .config import ModelConfig
from .orchestrator import SimulatorOrchestrator
from .sim.metrics import summary_table
from .simulations import simulate_agents

SLEEVE_AGENTS = ("ExternalPA", "ActiveExt", "InternalPA")
SUPPORTED_OBJECTIVES = ("total_return", "excess_return")
_StreamTuple = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


class _StreamCache:
    def __init__(self, cfg: ModelConfig, idx_series: pd.Series, seed: int | None) -> None:
        self._cfg = cfg
        self._idx_series = idx_series
        self._seed = seed
        self._streams: _StreamTuple | None = None

    def get(self) -> _StreamTuple | None:
        if self._streams is None:
            orch = SimulatorOrchestrator(self._cfg, self._idx_series)
            draw_streams = getattr(orch, "draw_streams", None)
            if draw_streams is None:
                return None
            self._streams = draw_streams(seed=self._seed)
        return self._streams


class _MinimizeResult(Protocol):
    x: np.ndarray
    success: bool
    message: object


_MinimizeFunc = Callable[..., _MinimizeResult]


def _clamp_grid(grid: np.ndarray, min_value: float | None, max_value: float | None) -> np.ndarray:
    min_val = 0.0 if min_value is None else min_value
    max_val = float("inf") if max_value is None else max_value
    return grid[(grid >= min_val) & (grid <= max_val)]


def _internal_cap_ok(value: float, min_internal: float | None, max_internal: float | None) -> bool:
    if min_internal is not None and value < min_internal:
        return False
    if max_internal is not None and value > max_internal:
        return False
    return True


def _pick_priority_combos(
    ext_grid: Iterable[float],
    act_grid: Iterable[float],
    total: float,
    min_internal: float | None,
    max_internal: float | None,
) -> list[tuple[float, float]]:
    ext_vals = sorted(set(ext_grid))
    act_vals = sorted(set(act_grid))
    if not ext_vals or not act_vals:
        return []

    candidates = [
        (ext_vals[0], act_vals[0]),
        (ext_vals[0], act_vals[-1]),
        (ext_vals[-1], act_vals[0]),
        (ext_vals[-1], act_vals[-1]),
    ]

    target = total / 3
    ext_mid = min(ext_vals, key=lambda v: abs(v - target))
    act_mid = min(act_vals, key=lambda v: abs(v - target))
    candidates.append((ext_mid, act_mid))

    picked: list[tuple[float, float]] = []
    for ext_cap, act_cap in candidates:
        int_cap = total - ext_cap - act_cap
        if int_cap < 0:
            continue
        if not _internal_cap_ok(int_cap, min_internal, max_internal):
            continue
        if (ext_cap, act_cap) not in picked:
            picked.append((ext_cap, act_cap))
    return picked


def _count_valid_combos(ext_vals: np.ndarray, act_vals: np.ndarray, total: float) -> int:
    if ext_vals.size == 0 or act_vals.size == 0:
        return 0
    count = 0
    min_act = float(act_vals[0])
    for ext in ext_vals:
        max_act = total - float(ext)
        if max_act < min_act:
            continue
        idx = int(np.searchsorted(act_vals, max_act, side="right"))
        count += idx
    return count


def _coerce_metric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)):
        try:
            metric = float(value)
        except ValueError:
            return None
    elif isinstance(value, (Real, np.floating, np.integer)):
        metric = float(value)
    else:
        return None
    if not np.isfinite(metric):
        return None
    return metric


def _objective_from_summary(summary: pd.DataFrame, objective: str) -> float | None:
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"objective must be one of: {', '.join(SUPPORTED_OBJECTIVES)}")
    column = "AnnReturn" if objective == "total_return" else "ExcessReturn"
    total_row = summary[summary["Agent"] == "Total"]
    if not total_row.empty:
        return _coerce_metric(total_row.iloc[0][column])
    values = []
    for agent in SLEEVE_AGENTS:
        sub = summary[summary["Agent"] == agent]
        if sub.empty:
            continue
        val = _coerce_metric(sub.iloc[0][column])
        if val is not None:
            values.append(val)
    if not values:
        return None
    return float(sum(values))


def _risk_score(metrics: dict[str, float], constraint_scope: str) -> float:
    score = 0.0
    if constraint_scope in {"sleeves", "both"}:
        for ag in SLEEVE_AGENTS:
            score += metrics.get(f"{ag}_TE", 0.0)
            score += metrics.get(f"{ag}_BreachProb", 0.0)
            score += abs(metrics.get(f"{ag}_CVaR", 0.0))
    if constraint_scope in {"total", "both"}:
        score += metrics.get("Total_TE", 0.0)
        score += metrics.get("Total_BreachProb", 0.0)
        score += abs(metrics.get("Total_CVaR", 0.0))
    return score


def _extract_metrics(
    summary: pd.DataFrame,
    *,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    constraint_scope: Literal["sleeves", "total", "both"],
) -> tuple[dict[str, float], bool] | None:
    meets = True
    metrics: dict[str, float] = {}
    for agent in SLEEVE_AGENTS:
        sub = summary[summary["Agent"] == agent]
        if sub.empty:
            continue
        row = sub.iloc[0]
        te = _coerce_metric(row["TE"])
        bprob = _coerce_metric(row["BreachProb"])
        cvar = _coerce_metric(row["CVaR"])
        if te is None or bprob is None or cvar is None:
            return None
        metrics[f"{agent}_TE"] = float(te)
        metrics[f"{agent}_BreachProb"] = float(bprob)
        metrics[f"{agent}_CVaR"] = float(cvar)
        if constraint_scope in {"sleeves", "both"} and (
            te > max_te or bprob > max_breach or abs(cvar) > max_cvar
        ):
            meets = False

    total_row = summary[summary["Agent"] == "Total"]
    if not total_row.empty:
        total_row = total_row.iloc[0]
        total_te = _coerce_metric(total_row["TE"])
        total_bprob = _coerce_metric(total_row["BreachProb"])
        total_cvar = _coerce_metric(total_row["CVaR"])
        if total_te is None or total_bprob is None or total_cvar is None:
            return None
        metrics.update(
            {
                "Total_TE": float(total_te),
                "Total_BreachProb": float(total_bprob),
                "Total_CVaR": float(total_cvar),
            }
        )
        if constraint_scope in {"total", "both"} and (
            total_te > max_te or total_bprob > max_breach or abs(total_cvar) > max_cvar
        ):
            meets = False
    return metrics, meets


def _evaluate_allocation(
    cfg: ModelConfig,
    idx_series: pd.Series,
    *,
    ext_cap: float,
    act_cap: float,
    int_cap: float,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    constraint_scope: Literal["sleeves", "total", "both"],
    seed: int | None,
    objective: str | None = None,
    stream_cache: _StreamCache | None = None,
) -> tuple[dict[str, float], bool, float | None] | None:
    test_cfg = cfg.model_copy(
        update={
            "external_pa_capital": float(ext_cap),
            "active_ext_capital": float(act_cap),
            "internal_pa_capital": float(int_cap),
        }
    )
    if stream_cache is None:
        streams = None
    else:
        streams = stream_cache.get()
    if streams is None:
        orch = SimulatorOrchestrator(test_cfg, idx_series)
        _, summary = orch.run(seed=seed)
    else:
        agents = build_from_config(test_cfg)
        returns = simulate_agents(agents, *streams)
        summary = summary_table(returns, benchmark="Base")
    result = _extract_metrics(
        summary,
        max_te=max_te,
        max_breach=max_breach,
        max_cvar=max_cvar,
        constraint_scope=constraint_scope,
    )
    if result is None:
        return None
    metrics, meets = result
    objective_value = None
    if objective is not None:
        objective_value = _objective_from_summary(summary, objective)
    return metrics, meets, objective_value


def _grid_sleeve_sizes(
    cfg: ModelConfig,
    idx_series: pd.Series,
    *,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    step: float,
    min_external: float | None,
    max_external: float | None,
    min_active: float | None,
    max_active: float | None,
    min_internal: float | None,
    max_internal: float | None,
    sort_by: str,
    seed: int | None,
    max_evals: int | None,
    constraint_scope: Literal["sleeves", "total", "both"],
    objective: str | None = None,
    stream_cache: _StreamCache | None = None,
) -> pd.DataFrame:
    if step <= 0:
        raise ValueError("step must be positive")

    total = cfg.total_fund_capital
    step_size = total * step
    grid = np.arange(0.0, total + 1e-9, step_size)
    ext_grid = _clamp_grid(grid, min_external, max_external)
    act_grid = _clamp_grid(grid, min_active, max_active)

    if ext_grid.size == 0 or act_grid.size == 0:
        return pd.DataFrame()

    combos: list[tuple[float, float]] | None = None
    if max_evals is not None:
        ext_vals = np.asarray(ext_grid, dtype=float)
        act_vals = np.asarray(act_grid, dtype=float)
        max_valid = _count_valid_combos(ext_vals, act_vals, total)
        if max_valid <= max_evals:
            combos = []
            for ext_cap in ext_vals:
                max_act = total - float(ext_cap)
                if max_act < act_vals[0]:
                    continue
                idx = int(np.searchsorted(act_vals, max_act, side="right"))
                for act_cap in act_vals[:idx]:
                    combos.append((float(ext_cap), float(act_cap)))
        else:
            rng = np.random.default_rng(seed)
            priority = _pick_priority_combos(
                ext_vals,
                act_vals,
                total,
                min_internal,
                max_internal,
            )
            if max_evals <= len(priority):
                combos = list(priority[:max_evals])
                seen = set(combos)
            else:
                combos = list(priority)
                seen = set(combos)
            attempts = 0
            max_attempts = max(1, max_evals) * 50
            while len(combos) < max_evals and attempts < max_attempts:
                attempts += 1
                ext_cap = float(ext_vals[rng.integers(0, len(ext_vals))])
                act_cap = float(act_vals[rng.integers(0, len(act_vals))])
                internal = total - ext_cap - act_cap
                if internal < 0:
                    continue
                if not _internal_cap_ok(internal, min_internal, max_internal):
                    continue
                pair = (ext_cap, act_cap)
                if pair in seen:
                    continue
                seen.add(pair)
                combos.append(pair)
            if len(combos) < max_evals:
                for ext_cap in ext_vals:
                    max_act = total - float(ext_cap)
                    if max_act < act_vals[0]:
                        continue
                    idx = int(np.searchsorted(act_vals, max_act, side="right"))
                    for act_cap in act_vals[:idx]:
                        pair = (float(ext_cap), float(act_cap))
                        internal = total - pair[0] - pair[1]
                        if not _internal_cap_ok(internal, min_internal, max_internal):
                            continue
                        if pair in seen:
                            continue
                        seen.add(pair)
                        combos.append(pair)
                        if len(combos) >= max_evals:
                            break
                    if len(combos) >= max_evals:
                        break

    records: list[dict[str, float]] = []

    def _maybe_record(ext_cap: float, act_cap: float) -> None:
        int_cap = total - ext_cap - act_cap

        # Bounds filtering (if provided)
        if min_external is not None and ext_cap < min_external:
            return
        if max_external is not None and ext_cap > max_external:
            return
        if min_active is not None and act_cap < min_active:
            return
        if max_active is not None and act_cap > max_active:
            return
        if min_internal is not None and int_cap < min_internal:
            return
        if max_internal is not None and int_cap > max_internal:
            return

        evaluated = _evaluate_allocation(
            cfg,
            idx_series,
            ext_cap=ext_cap,
            act_cap=act_cap,
            int_cap=int_cap,
            max_te=max_te,
            max_breach=max_breach,
            max_cvar=max_cvar,
            constraint_scope=constraint_scope,
            seed=seed,
            objective=objective,
            stream_cache=stream_cache,
        )
        if evaluated is None:
            return
        metrics, meets, objective_value = evaluated
        if meets:
            record = {
                "external_pa_capital": float(ext_cap),
                "active_ext_capital": float(act_cap),
                "internal_pa_capital": float(int_cap),
            }
            record.update(metrics)
            if objective_value is not None:
                record["objective_value"] = float(objective_value)
            record["risk_score"] = _risk_score(metrics, constraint_scope)
            records.append(record)

    if combos is None:
        for ext_cap in ext_grid:
            max_act = total - float(ext_cap)
            if max_act < act_grid[0]:
                continue
            for act_cap in act_grid:
                if float(act_cap) > max_act:
                    break
                _maybe_record(float(ext_cap), float(act_cap))
    else:
        for ext_cap, act_cap in combos:
            _maybe_record(ext_cap, act_cap)
    df = pd.DataFrame.from_records(records)
    if not df.empty and sort_by in df.columns:
        ascending = sort_by != "objective_value"
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return df


def _load_minimize() -> _MinimizeFunc | None:
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    return cast(_MinimizeFunc, minimize)


def _metric_slope(metric: float | None, capital: float, *, total: float) -> float | None:
    if metric is None:
        return None
    min_cap = max(total * 1e-4, 1e-6)
    denom = max(capital, min_cap)
    return metric / denom


def _build_linear_surrogates(
    cfg: ModelConfig,
    idx_series: pd.Series,
    *,
    seed: int | None,
    stream_cache: _StreamCache | None = None,
) -> dict[str, dict[str, float]] | None:
    if stream_cache is None:
        streams = None
    else:
        streams = stream_cache.get()
    if streams is None:
        orch = SimulatorOrchestrator(cfg, idx_series)
        _, summary = orch.run(seed=seed)
    else:
        agents = build_from_config(cfg)
        returns = simulate_agents(agents, *streams)
        summary = summary_table(returns, benchmark="Base")
    total = cfg.total_fund_capital
    slopes: dict[str, dict[str, float]] = {}
    for agent, capital in (
        ("ExternalPA", cfg.external_pa_capital),
        ("ActiveExt", cfg.active_ext_capital),
        ("InternalPA", cfg.internal_pa_capital),
    ):
        sub = summary[summary["Agent"] == agent]
        if sub.empty:
            return None
        row = sub.iloc[0]
        te = _coerce_metric(row["TE"])
        bprob = _coerce_metric(row["BreachProb"])
        cvar = _coerce_metric(row["CVaR"])
        ann_return = _coerce_metric(row["AnnReturn"])
        excess_return = _coerce_metric(row["ExcessReturn"])
        if (
            te is None
            or bprob is None
            or cvar is None
            or ann_return is None
            or excess_return is None
        ):
            return None
        slopes[agent] = {
            "TE": _metric_slope(te, capital, total=total) or 0.0,
            "BreachProb": _metric_slope(bprob, capital, total=total) or 0.0,
            "CVaR": _metric_slope(abs(cvar), capital, total=total) or 0.0,
            "AnnReturn": _metric_slope(ann_return, capital, total=total) or 0.0,
            "ExcessReturn": _metric_slope(excess_return, capital, total=total) or 0.0,
        }
    return slopes


def _initial_guess(
    cfg: ModelConfig,
    *,
    total: float,
    bounds: Sequence[tuple[float, float]],
    min_internal: float,
    max_internal: float,
) -> np.ndarray:
    ext_min, ext_max = bounds[0]
    act_min, act_max = bounds[1]
    ext = float(min(max(cfg.external_pa_capital, ext_min), ext_max))
    act = float(min(max(cfg.active_ext_capital, act_min), act_max))
    internal = total - ext - act
    if min_internal <= internal <= max_internal:
        return np.array([ext, act], dtype=float)
    target_internal = min(max(internal, min_internal), max_internal)
    remaining = total - target_internal
    min_sum = ext_min + act_min
    max_sum = ext_max + act_max
    remaining = min(max(remaining, min_sum), max_sum)
    ext_range = ext_max - ext_min
    act_range = act_max - act_min
    if ext_range + act_range == 0:
        return np.array([ext_min, act_min], dtype=float)
    ext = ext_min + (remaining - min_sum) * (ext_range / (ext_range + act_range))
    act = remaining - ext
    return np.array([ext, act], dtype=float)


def _optimize_sleeve_sizes(
    cfg: ModelConfig,
    idx_series: pd.Series,
    *,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    min_external: float | None,
    max_external: float | None,
    min_active: float | None,
    max_active: float | None,
    min_internal: float | None,
    max_internal: float | None,
    seed: int | None,
    constraint_scope: Literal["sleeves", "total", "both"],
    objective: str,
    maxiter: int,
    stream_cache: _StreamCache | None = None,
) -> tuple[pd.DataFrame | None, str]:
    minimize = _load_minimize()
    if minimize is None:
        return None, "missing_scipy"

    slopes = _build_linear_surrogates(cfg, idx_series, seed=seed, stream_cache=stream_cache)
    if slopes is None:
        return None, "missing_metrics"

    total = cfg.total_fund_capital
    min_ext = 0.0 if min_external is None else min_external
    max_ext = total if max_external is None else max_external
    min_act = 0.0 if min_active is None else min_active
    max_act = total if max_active is None else max_active
    min_int = 0.0 if min_internal is None else min_internal
    max_int = total if max_internal is None else max_internal

    bounds = [(min_ext, max_ext), (min_act, max_act)]
    x0 = _initial_guess(cfg, total=total, bounds=bounds, min_internal=min_int, max_internal=max_int)

    def internal_cap(x: np.ndarray) -> float:
        return float(total - x[0] - x[1])

    def cap_for_agent(x: np.ndarray, agent: str) -> float:
        if agent == "ExternalPA":
            return float(x[0])
        if agent == "ActiveExt":
            return float(x[1])
        return internal_cap(x)

    constraints: list[dict[str, object]] = []
    for agent in SLEEVE_AGENTS:
        if constraint_scope in {"sleeves", "both"}:
            for metric, limit in (("TE", max_te), ("BreachProb", max_breach), ("CVaR", max_cvar)):
                slope = slopes[agent].get(metric)
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, s=slope, lim=limit, ag=agent: lim
                        - s * cap_for_agent(x, ag),
                    }
                )

    if constraint_scope in {"total", "both"}:
        for metric, limit in (("TE", max_te), ("BreachProb", max_breach), ("CVaR", max_cvar)):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, lim=limit, met=metric: lim
                    - sum(slopes[ag][met] * cap_for_agent(x, ag) for ag in SLEEVE_AGENTS),
                }
            )

    constraints.append({"type": "ineq", "fun": lambda x, mn=min_int: internal_cap(x) - mn})
    constraints.append({"type": "ineq", "fun": lambda x, mx=max_int: mx - internal_cap(x)})

    if objective not in SUPPORTED_OBJECTIVES:
        return None, "invalid_objective"

    def objective_fn(x: np.ndarray) -> float:
        ext, act = float(x[0]), float(x[1])
        internal = internal_cap(x)
        key = "AnnReturn" if objective == "total_return" else "ExcessReturn"
        total_return = (
            slopes["ExternalPA"][key] * ext
            + slopes["ActiveExt"][key] * act
            + slopes["InternalPA"][key] * internal
        )
        return -float(total_return)

    result = minimize(
        objective_fn,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter},
    )

    ext_cap, act_cap = float(result.x[0]), float(result.x[1])
    int_cap = float(total - ext_cap - act_cap)
    evaluated = _evaluate_allocation(
        cfg,
        idx_series,
        ext_cap=ext_cap,
        act_cap=act_cap,
        int_cap=int_cap,
        max_te=max_te,
        max_breach=max_breach,
        max_cvar=max_cvar,
        constraint_scope=constraint_scope,
        seed=seed,
        objective=objective,
        stream_cache=stream_cache,
    )
    if evaluated is None:
        return None, "invalid_metrics"
    metrics, meets, objective_value = evaluated
    record: dict[str, object] = {
        "external_pa_capital": float(ext_cap),
        "active_ext_capital": float(act_cap),
        "internal_pa_capital": float(int_cap),
    }
    record.update(metrics)
    if objective_value is not None:
        record["objective_value"] = float(objective_value)
    record["risk_score"] = _risk_score(metrics, constraint_scope)
    record["constraints_satisfied"] = bool(meets)
    record["optimizer_success"] = bool(result.success)
    record["optimizer_status"] = str(result.message)
    record["objective"] = objective
    status = "optimizer_ok" if result.success else "optimizer_failed"
    return pd.DataFrame([record]), status


def suggest_sleeve_sizes(
    cfg: ModelConfig,
    idx_series: pd.Series,
    *,
    max_te: float,
    max_breach: float,
    max_cvar: float,
    step: float = 0.25,
    min_external: float | None = None,
    max_external: float | None = None,
    min_active: float | None = None,
    max_active: float | None = None,
    min_internal: float | None = None,
    max_internal: float | None = None,
    sort_by: str = "risk_score",
    seed: int | None = None,
    max_evals: int | None = 500,
    constraint_scope: Literal["sleeves", "total", "both"] = "sleeves",
    optimize: bool = False,
    objective: str = "total_return",
    optimizer_maxiter: int = 200,
) -> pd.DataFrame:
    """Suggest sleeve allocations that respect risk constraints.

    Performs a grid search over capital allocations for the three sleeves
    (external portable alpha, active extension and internal PA). When
    ``optimize`` is enabled, a coarse convex surrogate is used with
    ``scipy.optimize.minimize`` to target higher-return allocations, falling
    back to the grid if the optimizer fails.

    Parameters
    ----------
    cfg:
        Base :class:`~pa_core.config.ModelConfig` used as a template.
    idx_series:
        Benchmark return series used by :class:`SimulatorOrchestrator`.
    max_te:
        Maximum allowed tracking error per sleeve.
    max_breach:
        Maximum allowed breach probability per sleeve.
    max_cvar:
        Absolute CVaR cap per sleeve.
    step:
        Grid step as a fraction of ``total_fund_capital``.
    seed:
        Optional random seed for reproducibility.
    max_evals:
        If set and the Cartesian grid would exceed this number of
        combinations, a random subset of at most ``max_evals`` points is
        evaluated. This prevents exponential runtime as ``step`` becomes
        small.
    constraint_scope:
        Apply constraints to per-sleeve metrics, total portfolio metrics,
        or both.

    Returns
    -------
    pandas.DataFrame
        Table of feasible capital combinations and associated metrics.
    """

    if constraint_scope not in {"sleeves", "total", "both"}:
        raise ValueError("constraint_scope must be one of: sleeves, total, both")

    stream_cache = _StreamCache(cfg, idx_series, seed)

    if optimize:
        opt_df, status = _optimize_sleeve_sizes(
            cfg,
            idx_series,
            max_te=max_te,
            max_breach=max_breach,
            max_cvar=max_cvar,
            min_external=min_external,
            max_external=max_external,
            min_active=min_active,
            max_active=max_active,
            min_internal=min_internal,
            max_internal=max_internal,
            seed=seed,
            constraint_scope=constraint_scope,
            objective=objective,
            maxiter=optimizer_maxiter,
            stream_cache=stream_cache,
        )
        if opt_df is not None:
            meets = bool(opt_df.loc[0, "constraints_satisfied"])
            if bool(opt_df.loc[0, "optimizer_success"]) and meets:
                return opt_df.reset_index(drop=True)
        grid_df = _grid_sleeve_sizes(
            cfg,
            idx_series,
            max_te=max_te,
            max_breach=max_breach,
            max_cvar=max_cvar,
            step=step,
            min_external=min_external,
            max_external=max_external,
            min_active=min_active,
            max_active=max_active,
            min_internal=min_internal,
            max_internal=max_internal,
            sort_by="objective_value",
            seed=seed,
            max_evals=max_evals,
            constraint_scope=constraint_scope,
            objective=objective,
            stream_cache=stream_cache,
        )
        if grid_df.empty:
            if opt_df is None:
                return pd.DataFrame()
            opt_df["optimizer_status"] = f"fallback_failed:{status}"
            return opt_df.reset_index(drop=True)
        grid_df = grid_df.copy()
        grid_df["optimizer_status"] = f"grid_fallback:{status}"
        grid_df["optimizer_success"] = False
        grid_df["constraints_satisfied"] = True
        grid_df["objective"] = objective
        return grid_df.reset_index(drop=True)

    return _grid_sleeve_sizes(
        cfg,
        idx_series,
        max_te=max_te,
        max_breach=max_breach,
        max_cvar=max_cvar,
        step=step,
        min_external=min_external,
        max_external=max_external,
        min_active=min_active,
        max_active=max_active,
        min_internal=min_internal,
        max_internal=max_internal,
        sort_by=sort_by,
        seed=seed,
        max_evals=max_evals,
        constraint_scope=constraint_scope,
        objective=None,
        stream_cache=stream_cache,
    )
