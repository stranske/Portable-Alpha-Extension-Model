from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

import pandas as pd
from numpy.typing import NDArray

from ..backend import xp as np
from ..config import ModelConfig, normalize_share
from ..portfolio import compute_total_contribution_returns
from ..types import ArrayLike

__all__ = [
    "compute_sleeve_return_attribution",
    "compute_sleeve_return_contribution",
    "compute_sleeve_cvar_contribution",
    "compute_sleeve_risk_attribution",
]


def _cvar_tail_mask(total_arr: ArrayLike, confidence: float) -> tuple[float, NDArray[Any]]:
    """Return (VaR cutoff, tail mask) matching CVaR strict-tail semantics."""
    flat = np.asarray(total_arr, dtype=float).reshape(-1)
    var_cutoff = float(np.quantile(flat, 1 - confidence, method="lower"))
    tail_mask = flat < var_cutoff
    if not bool(np.any(tail_mask)):
        tail_mask = flat <= var_cutoff
    if not bool(np.any(tail_mask)):
        tail_mask = np.ones_like(flat, dtype=bool)
    return var_cutoff, tail_mask


def compute_sleeve_return_attribution(cfg: ModelConfig, idx_series: pd.Series) -> pd.DataFrame:
    """Compute per-agent monthly return attribution by component.

    Components:
    - Beta: exposure to index mean return
    - Alpha: exposure to sleeve-specific alpha stream mean
    - Financing: financing drag applied to beta portion where applicable

    Notes
    -----
    Uses the same monthly means as the simulator:
    - Index mean estimated from the provided ``idx_series`` (sample mean)
    - Sleeve alpha means sourced from ``cfg`` monthly parameters
    - Financing means taken directly from monthly financing parameters in ``cfg``
    """

    total = float(cfg.total_fund_capital)
    w_ext = float(cfg.external_pa_capital) / total if total > 0 else 0.0
    w_act = float(cfg.active_ext_capital) / total if total > 0 else 0.0
    w_int = float(cfg.internal_pa_capital) / total if total > 0 else 0.0
    leftover_beta = max(
        total - cfg.external_pa_capital - cfg.active_ext_capital - cfg.internal_pa_capital,
        0.0,
    )
    w_leftover = float(leftover_beta) / total if total > 0 else 0.0
    residual_label = "ResidualBeta"

    # Monthly means
    mu_idx_m = float(pd.Series(idx_series).mean())
    mu_H_m = float(cfg.mu_H)
    mu_E_m = float(cfg.mu_E)
    mu_M_m = float(cfg.mu_M)

    fin_int_m = float(cfg.internal_financing_mean_month)
    fin_ext_m = float(cfg.ext_pa_financing_mean_month)
    fin_act_m = float(cfg.act_ext_financing_mean_month)

    theta_extpa = normalize_share(getattr(cfg, "theta_extpa", 0.0)) or 0.0
    active_share = normalize_share(getattr(cfg, "active_share", 0.5)) or 0.0

    rows: List[Dict[str, object]] = []

    # Base (benchmark sleeve)
    base_beta = cfg.w_beta_H * mu_idx_m
    base_alpha = cfg.w_alpha_H * mu_H_m
    base_fin = -cfg.w_beta_H * fin_int_m
    rows += [
        {"Agent": "Base", "Sub": "Beta", "Return": base_beta},
        {"Agent": "Base", "Sub": "Alpha", "Return": base_alpha},
        {"Agent": "Base", "Sub": "Financing", "Return": base_fin},
    ]

    # ExternalPA
    if w_ext > 0:
        ext_beta = w_ext * mu_idx_m
        ext_alpha = w_ext * theta_extpa * mu_M_m
        ext_fin = -w_ext * fin_ext_m
        rows += [
            {"Agent": "ExternalPA", "Sub": "Beta", "Return": ext_beta},
            {"Agent": "ExternalPA", "Sub": "Alpha", "Return": ext_alpha},
            {"Agent": "ExternalPA", "Sub": "Financing", "Return": ext_fin},
        ]

    # ActiveExt
    if w_act > 0:
        act_beta = w_act * mu_idx_m
        act_alpha = w_act * active_share * mu_E_m
        act_fin = -w_act * fin_act_m
        rows += [
            {"Agent": "ActiveExt", "Sub": "Beta", "Return": act_beta},
            {"Agent": "ActiveExt", "Sub": "Alpha", "Return": act_alpha},
            {"Agent": "ActiveExt", "Sub": "Financing", "Return": act_fin},
        ]

    # InternalPA (pure alpha)
    if w_int > 0:
        int_alpha = w_int * mu_H_m
        rows.append({"Agent": "InternalPA", "Sub": "Alpha", "Return": int_alpha})

    # Attribution leftover beta (ResidualBeta) is not the simulation InternalBeta margin agent;
    # see docs/UserGuide.md "Sleeve Attribution Methodology" for InternalBeta vs UnexplainedBeta.
    if w_leftover > 0:
        ib_beta = w_leftover * mu_idx_m
        ib_fin = -w_leftover * fin_int_m
        rows += [
            {"Agent": residual_label, "Sub": "Beta", "Return": ib_beta},
            {"Agent": residual_label, "Sub": "Financing", "Return": ib_fin},
        ]

    df = pd.DataFrame(rows).reset_index(drop=True)
    if not df.empty:
        total_components = df[df["Agent"] != "Base"].groupby("Sub", as_index=False)["Return"].sum()
        if not total_components.empty:
            total_components["Agent"] = "Total"
            df = pd.concat(
                [df, total_components[["Agent", "Sub", "Return"]]],
                ignore_index=True,
            )
    return df


def compute_sleeve_return_contribution(
    returns_map: Mapping[str, ArrayLike],
    *,
    periods_per_year: int = 12,
    exclude: tuple[str, ...] = ("Base", "Total"),
) -> pd.DataFrame:
    """Compute per-sleeve return contribution to the total portfolio return.

    Contributions are arithmetic (mean monthly return * periods per year), so
    they sum to the total portfolio return within floating-point tolerance.
    """
    rows: List[Dict[str, object]] = []
    contributions_sum = 0.0
    total_returns = compute_total_contribution_returns(returns_map, exclude=exclude)
    total_value = None
    if total_returns is not None:
        total_arr = np.asarray(total_returns, dtype=float)
        total_value = float(total_arr.mean() * periods_per_year)

    for name, arr in returns_map.items():
        if name in exclude:
            continue
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.size == 0:
            contribution = 0.0
        else:
            contribution = float(arr_np.mean() * periods_per_year)
        contributions_sum += contribution
        rows.append({"Agent": name, "ReturnContribution": contribution})

    if rows:
        if total_value is None:
            total_value = contributions_sum
        rows.append({"Agent": "Total", "ReturnContribution": total_value})

    return pd.DataFrame(rows)


def compute_sleeve_cvar_contribution(
    returns_map: Mapping[str, ArrayLike],
    *,
    confidence: float = 0.95,
    exclude: tuple[str, ...] = ("Base", "Total"),
) -> pd.DataFrame:
    """Compute per-sleeve marginal CVaR contributions for the portfolio.

    Uses the conditional expectation of sleeve returns in the portfolio tail,
    so contributions sum to the portfolio CVaR (Euler decomposition).
    """
    expected_size = None
    for name, arr in returns_map.items():
        if name in exclude:
            continue
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.size == 0:
            continue
        if expected_size is None:
            expected_size = arr_np.size
        elif arr_np.size != expected_size:
            raise ValueError("sleeve returns must match total returns shape")

    total_returns = compute_total_contribution_returns(returns_map, exclude=exclude)
    if total_returns is None:
        return pd.DataFrame(columns=["Agent", "CVaRContribution"])

    total_arr = np.asarray(total_returns, dtype=float).reshape(-1)
    if total_arr.size == 0:
        return pd.DataFrame(columns=["Agent", "CVaRContribution"])

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    _, tail_mask = _cvar_tail_mask(total_arr, confidence)

    rows: List[Dict[str, object]] = []
    for name, arr in returns_map.items():
        if name in exclude:
            continue
        sleeve_arr = np.asarray(arr, dtype=float).reshape(-1)
        contribution = float(np.mean(sleeve_arr[tail_mask])) if sleeve_arr.size else 0.0
        rows.append({"Agent": name, "CVaRContribution": contribution})

    if rows:
        total_cvar = float(np.mean(total_arr[tail_mask]))
        rows.append({"Agent": "Total", "CVaRContribution": total_cvar})

    return pd.DataFrame(rows)


def compute_sleeve_risk_attribution(cfg: ModelConfig, idx_series: pd.Series) -> pd.DataFrame:
    """Approximate per-agent risk attribution and TE vs index.

    Provides simple, assumption-driven approximations using monthly moments:
    - BetaVol: beta exposure times index sigma (monthly, then annualised)
    - AlphaVol: alpha stream sigma scaled by sleeve intensity (monthly, then annualised)
    - CorrWithIndex: correlation of alpha stream with index (rho)
    - AnnVolApprox: sqrt(12) * stdev of agent return b*I + A using covariance
    - TEApprox: sqrt(12) * stdev of (agent - index)

    Notes: This is a heuristic decomposition for reporting that assumes
    configuration inputs are already in monthly units.
    """

    total = float(cfg.total_fund_capital)
    w_ext = float(cfg.external_pa_capital) / total if total > 0 else 0.0
    w_act = float(cfg.active_ext_capital) / total if total > 0 else 0.0
    w_int = float(cfg.internal_pa_capital) / total if total > 0 else 0.0
    leftover_beta = max(
        total - cfg.external_pa_capital - cfg.active_ext_capital - cfg.internal_pa_capital,
        0.0,
    )
    w_leftover = float(leftover_beta) / total if total > 0 else 0.0

    residual_label = "ResidualBeta"

    # Monthly sigmas (follow existing convention used in simulator params)
    idx_sigma_m = float(pd.Series(idx_series).std(ddof=1))
    sigma_H_m = float(cfg.sigma_H)
    sigma_E_m = float(cfg.sigma_E)
    sigma_M_m = float(cfg.sigma_M)

    theta_extpa = normalize_share(getattr(cfg, "theta_extpa", 0.0)) or 0.0
    active_share = normalize_share(getattr(cfg, "active_share", 0.5)) or 0.0

    def ann_vol(x_monthly: float) -> float:
        return math.sqrt(12.0) * x_monthly

    def _metrics(b: float, alpha_sigma: float, rho_idx_alpha: float) -> Dict[str, float]:
        # Monthly variances
        var_I = idx_sigma_m * idx_sigma_m
        var_A = alpha_sigma * alpha_sigma
        cov_IA = rho_idx_alpha * idx_sigma_m * alpha_sigma
        # Agent return R = b * I + A
        var_R = (b * b) * var_I + var_A + 2.0 * b * cov_IA
        # Tracking error: (R - I)
        d = b - 1.0
        var_TE = (d * d) * var_I + var_A + 2.0 * d * cov_IA
        # Correlation with index
        denom = (var_R * var_I) ** 0.5 if var_R > 0 and var_I > 0 else 0.0
        corr = (b * var_I + cov_IA) / denom if denom > 0 else 0.0
        return {
            "BetaVol": ann_vol(abs(b) * idx_sigma_m),
            "AlphaVol": ann_vol(alpha_sigma),
            "CorrWithIndex": float(corr),
            "AnnVolApprox": ann_vol(var_R**0.5),
            "TEApprox": ann_vol(var_TE**0.5),
        }

    rows: List[Dict[str, float | str]] = []
    # Base sleeve: beta and alpha from H
    rows.append(
        {
            "Agent": "Base",
            **_metrics(
                b=float(cfg.w_beta_H),
                alpha_sigma=float(cfg.w_alpha_H) * sigma_H_m,
                rho_idx_alpha=float(cfg.rho_idx_H),
            ),
        }
    )
    # ExternalPA
    if w_ext > 0:
        rows.append(
            {
                "Agent": "ExternalPA",
                **_metrics(
                    b=w_ext,
                    alpha_sigma=w_ext * theta_extpa * sigma_M_m,
                    rho_idx_alpha=float(cfg.rho_idx_M),
                ),
            }
        )
    # ActiveExt
    if w_act > 0:
        rows.append(
            {
                "Agent": "ActiveExt",
                **_metrics(
                    b=w_act,
                    alpha_sigma=w_act * active_share * sigma_E_m,
                    rho_idx_alpha=float(cfg.rho_idx_E),
                ),
            }
        )
    # InternalPA
    if w_int > 0:
        rows.append(
            {
                "Agent": "InternalPA",
                **_metrics(
                    b=0.0,
                    alpha_sigma=w_int * sigma_H_m,
                    rho_idx_alpha=float(cfg.rho_idx_H),
                ),
            }
        )
    # Attribution leftover beta (ResidualBeta) is not the simulation InternalBeta margin agent;
    # see docs/UserGuide.md "Sleeve Attribution Methodology" for InternalBeta vs UnexplainedBeta.
    if w_leftover > 0:
        rows.append(
            {
                "Agent": residual_label,
                **_metrics(
                    b=w_leftover,
                    alpha_sigma=0.0,
                    rho_idx_alpha=0.0,
                ),
            }
        )

    if total > 0:
        a_H = w_int * sigma_H_m
        a_E = w_act * active_share * sigma_E_m
        a_M = w_ext * theta_extpa * sigma_M_m
        var_alpha = (
            a_H * a_H
            + a_E * a_E
            + a_M * a_M
            + 2.0
            * (
                a_H * a_E * float(cfg.rho_H_E)
                + a_H * a_M * float(cfg.rho_H_M)
                + a_E * a_M * float(cfg.rho_E_M)
            )
        )
        alpha_sigma_total = math.sqrt(max(var_alpha, 0.0))
        cov_idx_alpha = idx_sigma_m * (
            a_H * float(cfg.rho_idx_H) + a_E * float(cfg.rho_idx_E) + a_M * float(cfg.rho_idx_M)
        )
        if alpha_sigma_total > 0 and idx_sigma_m > 0:
            rho_idx_alpha_total = cov_idx_alpha / (idx_sigma_m * alpha_sigma_total)
            rho_idx_alpha_total = max(-1.0, min(1.0, float(rho_idx_alpha_total)))
        else:
            rho_idx_alpha_total = 0.0
        total_beta = w_ext + w_act + w_leftover
        rows.append(
            {
                "Agent": "Total",
                **_metrics(
                    b=total_beta,
                    alpha_sigma=alpha_sigma_total,
                    rho_idx_alpha=rho_idx_alpha_total,
                ),
            }
        )

    return pd.DataFrame(rows)
