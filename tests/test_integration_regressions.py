from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pa_core.config import load_config, normalize_share
from pa_core.orchestrator import SimulatorOrchestrator
from pa_core.random import spawn_agent_rngs
from pa_core.reporting.attribution import (
    compute_sleeve_return_attribution,
    compute_sleeve_risk_attribution,
)
from pa_core.reporting.excel import export_to_excel
from pa_core.sim.params import build_simulation_params
from pa_core.sim.paths import draw_financing_series
from pa_core.validators import select_vol_regime_sigma

openpyxl = pytest.importorskip("openpyxl")
yaml = pytest.importorskip("yaml")


def test_integration_regression_active_ext_financing_export(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "params_template.yml"
    raw = yaml.safe_load(cfg_path.read_text())
    raw.update(
        {
            "N_SIMULATIONS": 100,
            "N_MONTHS": 12,
            "active_ext_capital": 50.0,
            "active_share": 0.5,
            "act_ext_financing_sigma_month": 0.01,
        }
    )
    cfg = load_config(raw)

    idx_series = pd.Series([0.01] * cfg.N_MONTHS)
    returns, summary = SimulatorOrchestrator(cfg, idx_series).run(seed=123)

    attr_df = compute_sleeve_return_attribution(cfg, idx_series)
    active_alpha = attr_df[(attr_df["Agent"] == "ActiveExt") & (attr_df["Sub"] == "Alpha")][
        "Return"
    ]
    # Regression guard for #1: ActiveExt alpha must be present and match the configured share/capital.
    assert not active_alpha.empty
    active_share = normalize_share(cfg.active_share) or 0.0
    expected_act_alpha = cfg.active_ext_capital / cfg.total_fund_capital * active_share * cfg.mu_E
    assert active_alpha.iloc[0] == pytest.approx(expected_act_alpha, rel=1e-6)

    mu_idx = float(idx_series.mean())
    idx_sigma, _, _ = select_vol_regime_sigma(
        idx_series,
        regime=cfg.vol_regime,
        window=cfg.vol_regime_window,
    )
    fin_params = build_simulation_params(cfg, mu_idx=mu_idx, idx_sigma=idx_sigma)
    fin_rngs = spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"])
    _f_int, _f_ext, f_act = draw_financing_series(
        n_months=cfg.N_MONTHS,
        n_sim=cfg.N_SIMULATIONS,
        params=fin_params,
        financing_mode=cfg.financing_mode,
        rngs=fin_rngs,
    )
    # Regression guard for #2: sigma > 0 should produce month-to-month variation.
    assert np.ptp(f_act[0]) > 0.0

    inputs_dict = {
        "_attribution_df": attr_df,
        "_risk_attr_df": compute_sleeve_risk_attribution(cfg, idx_series),
    }
    raw_returns_dict = {k: pd.DataFrame(v) for k, v in returns.items()}
    out_path = tmp_path / "integration.xlsx"
    export_to_excel(inputs_dict, summary, raw_returns_dict, filename=str(out_path))
    wb = openpyxl.load_workbook(out_path)
    # Regression guard for #3: export must include the Attribution sheet for reporting.
    assert {"Inputs", "Summary", "Attribution"} <= set(wb.sheetnames)
