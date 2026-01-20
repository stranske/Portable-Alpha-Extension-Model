from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd

from ..config import ModelConfig, normalize_share
from ..validators import calculate_margin_requirement

__all__ = ["build_agent_semantics"]

_TOLERANCE = 1e-6


def build_agent_semantics(cfg: ModelConfig) -> pd.DataFrame:
    """Return a DataFrame describing agent coefficient semantics."""
    columns = [
        "Agent",
        "capital_mm",
        "implied_capital_share",
        "beta_coeff_used",
        "alpha_coeff_used",
        "financing_coeff_used",
        "notes",
        "mismatch_flag",
    ]

    total_capital_value = getattr(cfg, "total_fund_capital", None)
    total_capital_missing = total_capital_value is None
    total_capital = float(total_capital_value or 0.0)
    if not math.isfinite(total_capital):
        total_capital = 0.0
    agents = list(_iter_agents(cfg)) if hasattr(cfg, "agents") else []
    if total_capital_missing and total_capital <= 0.0:
        summed_capital = 0.0
        for agent in agents:
            try:
                capital = float(agent["capital"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(capital):
                summed_capital += capital
        if summed_capital > 0.0:
            total_capital = summed_capital
    if not agents:
        return pd.DataFrame([], columns=columns)

    has_margin_inputs = all(
        hasattr(cfg, attr)
        for attr in (
            "reference_sigma",
            "volatility_multiple",
            "financing_model",
            "financing_schedule_path",
            "financing_term_months",
        )
    )
    if total_capital > 0.0:
        if has_margin_inputs:
            margin_requirement = calculate_margin_requirement(
                reference_sigma=cfg.reference_sigma,
                volatility_multiple=cfg.volatility_multiple,
                total_capital=total_capital,
                financing_model=cfg.financing_model,
                schedule_path=cfg.financing_schedule_path,
                term_months=cfg.financing_term_months,
            )
            if (
                math.isfinite(margin_requirement)
                and margin_requirement > 0.0
                and not any(a["name"] == "InternalBeta" for a in agents)
            ):
                agents.append(
                    {
                        "name": "InternalBeta",
                        "capital": margin_requirement,
                        "beta_share": margin_requirement / total_capital,
                        "alpha_share": 0.0,
                        "extra": {},
                    }
                )

    rows = [
        _build_row(
            agent["name"],
            agent["capital"],
            agent["beta_share"],
            agent["alpha_share"],
            agent.get("extra", {}),
            total_capital,
        )
        for agent in agents
    ]

    return pd.DataFrame(rows, columns=columns)


def _iter_agents(cfg: ModelConfig) -> Iterable[dict[str, Any]]:
    for agent in getattr(cfg, "agents", []):
        if isinstance(agent, dict):
            extra = agent.get("extra") or {}
            if not isinstance(extra, dict):
                extra = {}
            yield {
                "name": agent["name"],
                "capital": agent["capital"],
                "beta_share": agent["beta_share"],
                "alpha_share": agent["alpha_share"],
                "extra": extra,
            }
        else:
            extra = getattr(agent, "extra", {})
            if not isinstance(extra, dict):
                extra = {}
            yield {
                "name": agent.name,
                "capital": agent.capital,
                "beta_share": agent.beta_share,
                "alpha_share": agent.alpha_share,
                "extra": extra,
            }


def _build_row(
    name: str,
    capital: Any,
    beta_share: Any,
    alpha_share: Any,
    extra: dict[str, Any],
    total_capital: float,
) -> dict[str, Any]:
    def _coerce_float(value: Any, fallback: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(numeric):
            return fallback
        return numeric

    def _coerce_share(value: Any, fallback: float = 0.0) -> float:
        normalized = normalize_share(value)
        if normalized is None:
            return fallback
        try:
            return float(normalized)
        except (TypeError, ValueError):
            return fallback

    if not isinstance(extra, dict):
        extra = {}
    capital = _coerce_float(capital)
    beta_share = _coerce_share(beta_share)
    alpha_share = _coerce_share(alpha_share)

    implied_share = capital / total_capital if total_capital > 0.0 else 0.0
    notes = ""
    mismatch_flag = False

    if name == "Base":
        beta_coeff = beta_share
        alpha_coeff = alpha_share
        financing_coeff = -beta_share
    elif name == "ExternalPA":
        theta = _coerce_share(extra.get("theta_extpa", 0.0))
        beta_coeff = beta_share
        alpha_coeff = beta_share * theta
        financing_coeff = -beta_share
    elif name == "ActiveExt":
        active_share = _coerce_share(extra.get("active_share", 0.5))
        beta_coeff = beta_share
        alpha_coeff = beta_share * active_share
        financing_coeff = -beta_share
    elif name == "InternalPA":
        beta_coeff = 0.0
        alpha_coeff = alpha_share
        financing_coeff = 0.0
    elif name == "InternalBeta":
        beta_coeff = beta_share
        alpha_coeff = 0.0
        financing_coeff = -beta_share
    else:
        beta_coeff = beta_share
        alpha_coeff = alpha_share
        financing_coeff = -beta_share
        notes = "Semantics depend on the specific agent implementation"

    if total_capital <= 0.0:
        mismatch_flag = False
    elif name in {"ExternalPA", "ActiveExt", "InternalBeta"}:
        mismatch_flag = abs(implied_share - beta_share) > _TOLERANCE
    elif name == "InternalPA":
        mismatch_flag = abs(implied_share - alpha_share) > _TOLERANCE
    else:
        mismatch_flag = False

    return {
        "Agent": name,
        "capital_mm": capital,
        "implied_capital_share": implied_share,
        "beta_coeff_used": beta_coeff,
        "alpha_coeff_used": alpha_coeff,
        "financing_coeff_used": financing_coeff,
        "notes": notes,
        "mismatch_flag": mismatch_flag,
    }
