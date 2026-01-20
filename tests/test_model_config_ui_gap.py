import runpy


def _load_gap_module() -> dict:
    return runpy.run_path("scripts/model_config_ui_gap.py")


def test_gap_analysis_includes_expected_fields() -> None:
    module = _load_gap_module()
    gap = module["compute_gap"]()

    assert "backend" in gap["wired"]
    assert "N_SIMULATIONS" in gap["wired"]
    assert "financing_term_months" in gap["wired"]
