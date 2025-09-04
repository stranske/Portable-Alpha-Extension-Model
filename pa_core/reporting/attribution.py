from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd

from ..config import ModelConfig

__all__ = ["compute_sleeve_return_attribution", "compute_sleeve_risk_attribution"]


def compute_sleeve_return_attribution(
    cfg: ModelConfig, idx_series: pd.Series
) -> pd.DataFrame:
    """Compute per-agent annual return attribution by component.

    Components:
    - Beta: exposure to index mean return
    - Alpha: exposure to sleeve-specific alpha stream mean
    - Financing: financing drag applied to beta portion where applicable

    Notes
    -----
    Uses the same monthly means as the simulator:
    - Index mean estimated from the provided ``idx_series`` (sample mean)
    - Sleeve alpha means derived from ``cfg`` annual parameters divided by 12
    - Financing means taken directly from monthly financing parameters in ``cfg``
    """

    total = float(cfg.total_fund_capital)
    w_ext = float(cfg.external_pa_capital) / total if total > 0 else 0.0
    w_act = float(cfg.active_ext_capital) / total if total > 0 else 0.0
    w_int = float(cfg.internal_pa_capital) / total if total > 0 else 0.0
    leftover_beta = max(
        total
        - cfg.external_pa_capital
        - cfg.active_ext_capital
        - cfg.internal_pa_capital,
        0.0,
    )
    w_leftover = float(leftover_beta) / total if total > 0 else 0.0

    # Monthly means
    mu_idx_m = float(pd.Series(idx_series).mean())
    mu_H_m = float(cfg.mu_H) / 12.0
    mu_E_m = float(cfg.mu_E) / 12.0
    mu_M_m = float(cfg.mu_M) / 12.0

    fin_int_m = float(cfg.internal_financing_mean_month)
    fin_ext_m = float(cfg.ext_pa_financing_mean_month)
    fin_act_m = float(cfg.act_ext_financing_mean_month)

    theta_extpa = float(getattr(cfg, "theta_extpa", 0.0))
    active_share = float(getattr(cfg, "active_share", 50.0)) / 100.0

    def annual(x: float) -> float:
        return 12.0 * x

    rows: List[Dict[str, object]] = []

    # Base (benchmark sleeve)
    base_beta = annual(cfg.w_beta_H * mu_idx_m)
    base_alpha = annual(cfg.w_alpha_H * mu_H_m)
    base_fin = annual(-cfg.w_beta_H * fin_int_m)
    rows += [
        {"Agent": "Base", "Sub": "Beta", "Return": base_beta},
        {"Agent": "Base", "Sub": "Alpha", "Return": base_alpha},
        {"Agent": "Base", "Sub": "Financing", "Return": base_fin},
    ]

    # ExternalPA
    if w_ext > 0:
        ext_beta = annual(w_ext * mu_idx_m)
        ext_alpha = annual(w_ext * theta_extpa * mu_M_m)
        ext_fin = annual(-w_ext * fin_ext_m)
        rows += [
            {"Agent": "ExternalPA", "Sub": "Beta", "Return": ext_beta},
            {"Agent": "ExternalPA", "Sub": "Alpha", "Return": ext_alpha},
            {"Agent": "ExternalPA", "Sub": "Financing", "Return": ext_fin},
        ]

    # ActiveExt
    if w_act > 0:
        act_beta = annual(w_act * mu_idx_m)
        act_alpha = annual(w_act * active_share * mu_E_m)
        act_fin = annual(-w_act * fin_act_m)
        rows += [
            {"Agent": "ActiveExt", "Sub": "Beta", "Return": act_beta},
            {"Agent": "ActiveExt", "Sub": "Alpha", "Return": act_alpha},
            {"Agent": "ActiveExt", "Sub": "Financing", "Return": act_fin},
        ]

    # InternalPA (pure alpha)
    if w_int > 0:
        int_alpha = annual(w_int * mu_H_m)
        rows.append({"Agent": "InternalPA", "Sub": "Alpha", "Return": int_alpha})

    # InternalBeta (leftover beta)
    if w_leftover > 0:
        ib_beta = annual(w_leftover * mu_idx_m)
        ib_fin = annual(-w_leftover * fin_int_m)
        rows += [
            {"Agent": "InternalBeta", "Sub": "Beta", "Return": ib_beta},
            {"Agent": "InternalBeta", "Sub": "Financing", "Return": ib_fin},
        ]

    df = pd.DataFrame(rows).reset_index(drop=True)
    return df


def compute_sleeve_risk_attribution(
    cfg: ModelConfig, idx_series: pd.Series
) -> pd.DataFrame:
    """Approximate per-agent risk attribution and TE vs index.

    Provides simple, assumption-driven approximations using monthly moments:
    - BetaVol: beta exposure times index sigma (monthly, then annualised)
    - AlphaVol: alpha stream sigma scaled by sleeve intensity (monthly, then annualised)
    - CorrWithIndex: correlation of alpha stream with index (rho)
    - AnnVolApprox: sqrt(12) * stdev of agent return b*I + A using covariance
    - TEApprox: sqrt(12) * stdev of (agent - index)

    Notes: This is a heuristic decomposition for reporting, aligned with
    the simulator's convention of dividing annual sigma by 12 for monthly.
    """

    total = float(cfg.total_fund_capital)
    w_ext = float(cfg.external_pa_capital) / total if total > 0 else 0.0
    w_act = float(cfg.active_ext_capital) / total if total > 0 else 0.0
    w_int = float(cfg.internal_pa_capital) / total if total > 0 else 0.0
    leftover_beta = max(
        total
        - cfg.external_pa_capital
        - cfg.active_ext_capital
        - cfg.internal_pa_capital,
        0.0,
    )
    w_leftover = float(leftover_beta) / total if total > 0 else 0.0

    # Monthly sigmas (follow existing convention used in simulator params)
    idx_sigma_m = float(pd.Series(idx_series).std(ddof=1))
    sigma_H_m = float(cfg.sigma_H) / 12.0
    sigma_E_m = float(cfg.sigma_E) / 12.0
    sigma_M_m = float(cfg.sigma_M) / 12.0

    theta_extpa = float(getattr(cfg, "theta_extpa", 0.0))
    active_share = float(getattr(cfg, "active_share", 50.0)) / 100.0

    def ann_vol(x_monthly: float) -> float:
        return math.sqrt(12.0) * x_monthly

    def _metrics(
        b: float, alpha_sigma: float, rho_idx_alpha: float
    ) -> Dict[str, float]:
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
    # InternalBeta (leftover)
    if w_leftover > 0:
        rows.append(
            {
                "Agent": "InternalBeta",
                **_metrics(
                    b=w_leftover,
                    alpha_sigma=0.0,
                    rho_idx_alpha=0.0,
                ),
            }
        )

    return pd.DataFrame(rows)
