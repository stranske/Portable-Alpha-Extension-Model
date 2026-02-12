from __future__ import annotations

import runpy

from pa_core.config import load_config
from pa_core.llm.config_patch import apply_patch, validate_patch_dict
from pa_core.wizard_schema import AnalysisMode, get_default_config


def test_patch_roundtrip_yaml_reload_preserves_key_fields() -> None:
    module = runpy.run_path("dashboard/pages/3_Scenario_Wizard.py")
    build_yaml_from_config = module["_build_yaml_from_config"]

    config = get_default_config(AnalysisMode.RETURNS)
    patch = validate_patch_dict(
        {
            "set": {
                "analysis_mode": "returns",
                "n_simulations": 5000,
                "total_fund_capital": 1500.0,
                "sleeve_max_breach": 0.15,
            },
            "merge": {},
            "remove": [],
        }
    )

    applied = apply_patch(config, patch)
    yaml_dict = build_yaml_from_config(applied)
    loaded = load_config(yaml_dict)

    assert loaded.analysis_mode == applied.analysis_mode.value
    assert loaded.N_SIMULATIONS == applied.n_simulations
    assert loaded.total_fund_capital == applied.total_fund_capital
    assert loaded.sleeve_max_breach == applied.sleeve_max_breach
