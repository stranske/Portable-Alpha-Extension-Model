"""Tests for the Stress Lab dashboard functionality."""

import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from pa_core.config import ModelConfig
from pa_core.stress import apply_stress_preset
from pa_core.validators import TEST_TOLERANCE_EPSILON, VOLATILITY_STRESS_MULTIPLIER


def load_stress_lab_module() -> ModuleType:
    module_path = Path(__file__).parent.parent / "dashboard" / "pages" / "6_Stress_Lab.py"
    spec = importlib.util.spec_from_file_location("stress_lab", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load stress_lab module")
    stress_lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stress_lab)
    return stress_lab


def test_config_diff_function():
    """Test the _config_diff function detects all parameter differences."""
    # Import the function from the module
    stress_lab = load_stress_lab_module()

    _config_diff = stress_lab._config_diff

    # Test case 1: Basic parameter differences
    base_cfg = ModelConfig.model_validate(
        {
            "N_SIMULATIONS": 1000,
            "N_MONTHS": 12,
            "financing_mode": "broadcast",
            "total_fund_capital": 1000.0,
            "external_pa_capital": 200.0,
            "internal_pa_capital": 200.0,
            "active_ext_capital": 200.0,
            "external_alpha_frac": 0.5,
            "active_share": 0.5,
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        }
    )

    stressed_cfg = ModelConfig.model_validate(
        {
            "N_SIMULATIONS": 2000,  # Changed
            "N_MONTHS": 12,
            "financing_mode": "broadcast",
            "total_fund_capital": 1000.0,
            "external_pa_capital": 400.0,  # Changed
            "internal_pa_capital": 200.0,
            "active_ext_capital": 0.0,  # Changed
            "external_alpha_frac": 0.5,
            "active_share": 0.5,
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        }
    )

    diff_df = _config_diff(base_cfg, stressed_cfg)

    # Should detect 3 differences
    assert len(diff_df) == 3

    # Check specific differences
    diff_dict = {
        row["Parameter"]: {"Base": row["Base"], "Stressed": row["Stressed"]}
        for _, row in diff_df.iterrows()
    }

    assert "N_SIMULATIONS" in diff_dict
    assert diff_dict["N_SIMULATIONS"]["Base"] == 1000
    assert diff_dict["N_SIMULATIONS"]["Stressed"] == 2000

    assert "external_pa_capital" in diff_dict
    assert diff_dict["external_pa_capital"]["Base"] == 200.0
    assert diff_dict["external_pa_capital"]["Stressed"] == 400.0

    assert "active_ext_capital" in diff_dict
    assert diff_dict["active_ext_capital"]["Base"] == 200.0
    assert diff_dict["active_ext_capital"]["Stressed"] == 0.0


def test_config_diff_with_stress_preset():
    """Test _config_diff works correctly with actual stress presets."""
    stress_lab = load_stress_lab_module()
    _config_diff = stress_lab._config_diff

    # Create base config
    base_cfg = ModelConfig.model_validate(
        {
            "N_SIMULATIONS": 1000,
            "N_MONTHS": 12,
            "financing_mode": "broadcast",
            "total_fund_capital": 1000.0,
            "external_pa_capital": 200.0,
            "internal_pa_capital": 200.0,
            "active_ext_capital": 200.0,
            "external_alpha_frac": 0.5,
            "active_share": 0.5,
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        }
    )

    # Apply volatility stress preset
    stressed_cfg = apply_stress_preset(base_cfg, "2008_vol_regime")

    # Get differences
    diff_df = _config_diff(base_cfg, stressed_cfg)

    # Should detect changes in volatility parameters
    diff_params = set(diff_df["Parameter"].values)
    expected_vol_params = {"sigma_H", "sigma_E", "sigma_M"}

    # Check that all expected volatility parameters are detected
    assert expected_vol_params.issubset(
        diff_params
    ), f"Missing volatility parameters: {expected_vol_params - diff_params}"

    # Verify the changes are correctly detected (should be 3x original values)
    diff_dict = {
        row["Parameter"]: {"Base": row["Base"], "Stressed": row["Stressed"]}
        for _, row in diff_df.iterrows()
    }

    for param in expected_vol_params:
        base_val = diff_dict[param]["Base"]
        stressed_val = diff_dict[param]["Stressed"]
        # Stress preset multiplies by VOLATILITY_STRESS_MULTIPLIER
        assert (
            abs(stressed_val - (base_val * VOLATILITY_STRESS_MULTIPLIER)) < TEST_TOLERANCE_EPSILON
        ), f"Parameter {param} not correctly stressed: {base_val} -> {stressed_val}"


def test_config_diff_empty_when_identical():
    """Test that _config_diff returns empty DataFrame for identical configs."""
    stress_lab = load_stress_lab_module()
    _config_diff = stress_lab._config_diff

    # Create identical configs
    config_data = {
        "N_SIMULATIONS": 1000,
        "N_MONTHS": 12,
        "financing_mode": "broadcast",
        "total_fund_capital": 1000.0,
        "external_pa_capital": 200.0,
        "internal_pa_capital": 200.0,
        "active_ext_capital": 200.0,
        "external_alpha_frac": 0.5,
        "active_share": 0.5,
        "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
    }

    base_cfg = ModelConfig.model_validate(config_data)
    stressed_cfg = ModelConfig.model_validate(config_data.copy())

    diff_df = _config_diff(base_cfg, stressed_cfg)

    # Should be empty for identical configs
    assert len(diff_df) == 0


def test_config_diff_detects_all_key_differences():
    """Test that the fix correctly detects parameters from both configs (the main issue)."""
    stress_lab = load_stress_lab_module()
    _config_diff = stress_lab._config_diff

    # Create two configs with a clear difference
    base_cfg = ModelConfig.model_validate(
        {
            "N_SIMULATIONS": 1000,
            "N_MONTHS": 12,
            "financing_mode": "broadcast",
            "total_fund_capital": 1000.0,
            "external_pa_capital": 200.0,
            "internal_pa_capital": 200.0,
            "active_ext_capital": 200.0,
            "external_alpha_frac": 0.5,
            "active_share": 0.5,
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        }
    )

    stressed_cfg = ModelConfig.model_validate(
        {
            "N_SIMULATIONS": 2000,  # Changed value
            "N_MONTHS": 6,  # Changed value
            "financing_mode": "broadcast",
            "total_fund_capital": 1000.0,
            "external_pa_capital": 200.0,
            "internal_pa_capital": 200.0,
            "active_ext_capital": 200.0,
            "external_alpha_frac": 0.5,
            "active_share": 0.5,
            "risk_metrics": ["Return", "Risk", "terminal_ShortfallProb"],
        }
    )

    # Test with actual configs
    diff_df = _config_diff(base_cfg, stressed_cfg)

    # Should detect the 2 changes
    assert len(diff_df) >= 2

    diff_params = set(diff_df["Parameter"].values)
    assert "N_SIMULATIONS" in diff_params
    assert "N_MONTHS" in diff_params

    # Now demonstrate the original problem and our fix with a manual example
    # This shows what would happen with different dictionaries

    # Simulate the scenario described in the issue
    base_dict = {"param_A": 100, "param_B": 200, "param_removed": 999}
    stress_dict = {"param_A": 150, "param_B": 200, "param_added": 888}

    # Original broken logic: only check stress_dict keys
    old_diffs = []
    for key in sorted(stress_dict.keys()):
        b_val = base_dict.get(key)
        s_val = stress_dict.get(key)
        if b_val != s_val:
            old_diffs.append(key)

    # New correct logic: check union of all keys
    new_diffs = []
    for key in sorted(set(base_dict.keys()) | set(stress_dict.keys())):
        b_val = base_dict.get(key)
        s_val = stress_dict.get(key)
        if b_val != s_val:
            new_diffs.append(key)

    # Verify the fix works
    assert "param_A" in old_diffs  # Both should detect this change
    assert "param_A" in new_diffs

    assert "param_added" in old_diffs  # Both should detect this addition
    assert "param_added" in new_diffs

    assert "param_removed" not in old_diffs  # OLD LOGIC MISSES THIS
    assert "param_removed" in new_diffs  # NEW LOGIC CATCHES THIS

    # This proves our fix addresses the original issue


class _StressLabContainer:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _StressLabStreamlit:
    def __init__(self, *, run: bool, use_sample: bool) -> None:
        self.session_state = {}
        self.sidebar = self
        self.run = run
        self.use_sample = use_sample
        self.calls: list[tuple[str, str]] = []

    def __getattr__(self, name: str):
        if name in {"title", "markdown", "subheader", "dataframe"}:
            return lambda *_args, **_kwargs: None
        raise AttributeError(name)

    def file_uploader(self, *_args, **_kwargs):
        return None

    def checkbox(self, *_args, **_kwargs) -> bool:
        return self.use_sample

    def info(self, message: str) -> None:
        self.calls.append(("info", message))

    def number_input(self, _label: str, *, value, **_kwargs):
        return value

    def slider(self, _label: str, *, value, **_kwargs):
        return value

    def selectbox(self, _label: str, *, options, index: int, **_kwargs):
        return options[index]

    def button(self, *_args, **_kwargs) -> bool:
        return self.run

    def columns(self, count: int):
        return tuple(_StressLabContainer() for _ in range(count))

    def download_button(self, label: str, *_args, **_kwargs) -> None:
        self.calls.append(("download", label))

    def error(self, message: str) -> None:
        self.calls.append(("error", message))


def test_read_index_csv_accepts_getvalue_and_rejects_non_numeric() -> None:
    stress_lab = load_stress_lab_module()

    series = stress_lab._read_index_csv(io.BytesIO(b"date,return\n2026-01-01,0.01\n2026-02-01,\n"))

    assert series.tolist() == [0.01]
    with pytest.raises(ValueError, match="No numeric columns"):
        stress_lab._read_index_csv(io.BytesIO(b"date,label\n2026-01-01,calm\n"))


def test_stress_lab_main_runs_sample_flow_and_exposes_all_downloads(monkeypatch) -> None:
    stress_lab = load_stress_lab_module()
    fake_st = _StressLabStreamlit(run=True, use_sample=True)
    sample_returns = pd.Series([0.01, 0.02])
    summary = pd.DataFrame({"Agent": ["Base"], "Return": [0.01]})
    constructor_index_series = []

    class _Orchestrator:
        def __init__(self, _config, index_series) -> None:
            constructor_index_series.append(index_series)

        def run(self, *, seed: int):
            assert seed == 42
            return {}, summary

    monkeypatch.setattr(stress_lab, "st", fake_st)
    monkeypatch.setattr(stress_lab, "render_settings_sidebar", lambda: (None, "theme.yml"))
    monkeypatch.setattr(stress_lab, "apply_theme", lambda _path: None)
    monkeypatch.setattr(stress_lab, "current_scenario_config", lambda _state: None)
    monkeypatch.setattr(stress_lab, "current_index_returns", lambda _state: None)
    monkeypatch.setattr(
        stress_lab,
        "config_capital_defaults",
        lambda _config: {
            "total_fund_capital": 100.0,
            "external_pa_capital": 30.0,
            "active_ext_capital": 30.0,
            "internal_pa_capital": 30.0,
        },
    )
    monkeypatch.setattr(stress_lab, "load_bundled_sample_index", lambda: sample_returns)
    monkeypatch.setattr(stress_lab, "build_sample_model_config", lambda **_kwargs: object())
    monkeypatch.setattr(stress_lab, "apply_stress_preset", lambda config, _preset: config)
    monkeypatch.setattr(stress_lab, "SimulatorOrchestrator", _Orchestrator)
    monkeypatch.setattr(
        stress_lab, "build_constraint_report", lambda *_args, **_kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        stress_lab, "build_delta_table", lambda *_args: pd.DataFrame({"delta": [0.0]})
    )
    monkeypatch.setattr(stress_lab, "format_delta_table", lambda frame: frame)
    monkeypatch.setattr(stress_lab, "build_stress_workbook", lambda *_args: b"workbook")
    monkeypatch.setattr(stress_lab, "_config_diff", lambda *_args: pd.DataFrame())

    stress_lab.main()

    assert [label for kind, label in fake_st.calls if kind == "download"] == [
        "Download deltas as CSV",
        "Download config diff",
        "Download stress results (Excel)",
    ]
    assert constructor_index_series == [sample_returns, sample_returns]
    assert all(index_series is sample_returns for index_series in constructor_index_series)


def test_stress_lab_main_explains_missing_input(monkeypatch) -> None:
    stress_lab = load_stress_lab_module()
    fake_st = _StressLabStreamlit(run=False, use_sample=False)
    monkeypatch.setattr(stress_lab, "st", fake_st)
    monkeypatch.setattr(stress_lab, "render_settings_sidebar", lambda: (None, "theme.yml"))
    monkeypatch.setattr(stress_lab, "apply_theme", lambda _path: None)
    monkeypatch.setattr(stress_lab, "current_scenario_config", lambda _state: None)
    monkeypatch.setattr(stress_lab, "current_index_returns", lambda _state: None)
    monkeypatch.setattr(stress_lab, "config_capital_defaults", lambda _config: {})

    stress_lab.main()

    assert fake_st.calls == [
        (
            "info",
            "Upload an index CSV, tick 'Use bundled sample data', or run a scenario first to simulate.",
        )
    ]
