from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pa_core.config import load_config
from pa_core.orchestrator import SimulatorOrchestrator
from pa_core.random import spawn_agent_rngs
from pa_core.reporting.attribution import (
    compute_sleeve_return_attribution,
    compute_sleeve_risk_attribution,
)
from pa_core.reporting.excel import export_to_excel
from pa_core.sim.paths import draw_financing_series

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
    # Regression guard for #1: ActiveExt alpha must be nonzero when capital and active_share > 0.
    assert not active_alpha.empty
    assert active_alpha.iloc[0] > 1e-6

    fin_params = {
        "internal_financing_mean_month": cfg.internal_financing_mean_month,
        "internal_financing_sigma_month": cfg.internal_financing_sigma_month,
        "internal_spike_prob": cfg.internal_spike_prob,
        "internal_spike_factor": cfg.internal_spike_factor,
        "ext_pa_financing_mean_month": cfg.ext_pa_financing_mean_month,
        "ext_pa_financing_sigma_month": cfg.ext_pa_financing_sigma_month,
        "ext_pa_spike_prob": cfg.ext_pa_spike_prob,
        "ext_pa_spike_factor": cfg.ext_pa_spike_factor,
        "act_ext_financing_mean_month": cfg.act_ext_financing_mean_month,
        "act_ext_financing_sigma_month": cfg.act_ext_financing_sigma_month,
        "act_ext_spike_prob": cfg.act_ext_spike_prob,
        "act_ext_spike_factor": cfg.act_ext_spike_factor,
    }
    fin_rngs = spawn_agent_rngs(123, ["internal", "external_pa", "active_ext"])
    _f_int, _f_ext, f_act = draw_financing_series(
        n_months=cfg.N_MONTHS,
        n_sim=cfg.N_SIMULATIONS,
        params=fin_params,
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
