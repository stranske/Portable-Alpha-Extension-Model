from __future__ import annotations

from typing import List, Dict

import pandas as pd

from ..config import ModelConfig

__all__ = ["compute_sleeve_return_attribution"]


def compute_sleeve_return_attribution(cfg: ModelConfig, idx_series: pd.Series) -> pd.DataFrame:
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
    leftover_beta = max(total - cfg.external_pa_capital - cfg.active_ext_capital - cfg.internal_pa_capital, 0.0)
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

    return df.reset_index(drop=True)
