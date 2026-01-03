from scripts import settings_effectiveness as se


def test_settings_effectiveness_detects_output_change() -> None:
    base_cfg = se.build_base_config(n_simulations=100, n_months=6)
    base_summary = se.run_summary(base_cfg, seed=42)
    updated_cfg = base_cfg.model_copy(update={"mu_H": 0.06})
    updated_summary = se.run_summary(updated_cfg, seed=42)
    assert se.summary_changed(base_summary, updated_summary, tol=1e-9)


def test_settings_effectiveness_inputs_change() -> None:
    base_cfg = se.build_base_config(n_simulations=100, n_months=4)
    updated_cfg = base_cfg.model_copy(
        update={"risk_metrics": ["Return", "Risk", "terminal_ShortfallProb", "monthly_BreachProb"]}
    )
    base_inputs = se.run_inputs(base_cfg, seed=7)
    updated_inputs = se.run_inputs(updated_cfg, seed=7)
    assert se.inputs_changed(base_inputs, updated_inputs, ["risk_metrics"])
