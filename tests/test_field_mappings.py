"""Test field mappings functionality."""

import csv
import os
import tempfile
from pathlib import Path

from pa_core.config import ModelConfig, get_field_mappings


def test_get_field_mappings():
    """Test that get_field_mappings returns consistent mappings."""
    mappings = get_field_mappings()

    # Test that all expected core mappings are present
    expected_core_mappings = {
        "Analysis mode": "analysis_mode",
        "Number of simulations": "N_SIMULATIONS",
        "Number of months": "N_MONTHS",
        "External PA capital (mm)": "external_pa_capital",
        "Active Extension capital (mm)": "active_ext_capital",
        "Internal PA capital (mm)": "internal_pa_capital",
        "Total fund capital (mm)": "total_fund_capital",
        "risk_metrics": "risk_metrics",
    }

    for alias, field_name in expected_core_mappings.items():
        assert alias in mappings, f"Missing mapping for '{alias}'"
        assert (
            mappings[alias] == field_name
        ), f"Wrong mapping for '{alias}': expected '{field_name}', got '{mappings[alias]}'"


def test_field_mappings_consistency():
    """Test that field mappings are consistent with ModelConfig field definitions."""
    mappings = get_field_mappings()
    model_fields = ModelConfig.model_fields

    # Every mapped field should exist in the model
    for alias, field_name in mappings.items():
        assert (
            field_name in model_fields
        ), f"Field '{field_name}' mapped from alias '{alias}' does not exist in ModelConfig"


def test_csv_conversion_with_field_mappings():
    """Test CSV to YAML conversion using the new field mappings."""
    from pa_core.pa import _convert_csv_to_yaml

    # Create a test CSV file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Analysis mode", "returns"])
        writer.writerow(["Number of simulations", "100"])
        writer.writerow(["Number of months", "12"])
        writer.writerow(["External PA capital (mm)", "500.0"])
        writer.writerow(["Total fund capital (mm)", "1000.0"])
        writer.writerow(["In-House annual return (%)", "4.0"])
        writer.writerow(["risk_metrics", "Return;Risk;ShortfallProb"])
        csv_path = f.name

    try:
        # Create temporary YAML file for output
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yml") as yaml_f:
            yaml_path = yaml_f.name

        try:
            # Convert to YAML
            _convert_csv_to_yaml(csv_path, yaml_path)

            # Verify the YAML file was created
            assert Path(yaml_path).exists(), "YAML file was not created"

            # Load and verify the converted configuration
            from pa_core.config import load_config

            config = load_config(yaml_path)

            # Verify some key values
            assert config.analysis_mode == "returns"
            assert config.N_SIMULATIONS == 100
            assert config.N_MONTHS == 12
            assert config.external_pa_capital == 500.0
            assert config.total_fund_capital == 1000.0
            assert config.mu_H == 0.04  # Should be converted from percentage
            assert "Return" in config.risk_metrics
            assert "Risk" in config.risk_metrics
            assert "ShortfallProb" in config.risk_metrics
        finally:
            os.remove(yaml_path)
    finally:
        os.remove(csv_path)


def test_legacy_compatibility():
    """Test that the new approach maintains compatibility with existing YAML configs."""
    from pa_core.config import load_config

    # Test loading with snake_case field names (current YAML format)
    config_data = {
        "N_SIMULATIONS": 100,
        "N_MONTHS": 12,
        "analysis_mode": "returns",
        "external_pa_capital": 500.0,
        "mu_H": 0.04,
        "risk_metrics": ["Return", "Risk", "ShortfallProb"],
    }

    config = load_config(config_data)
    assert config.N_SIMULATIONS == 100
    assert config.analysis_mode == "returns"
    assert config.external_pa_capital == 500.0


def test_alias_compatibility():
    """Test that the new aliases work for loading configurations."""
    from pa_core.config import load_config

    # Test loading with human-readable aliases
    config_data = {
        "Number of simulations": 100,
        "Number of months": 12,
        "Analysis mode": "returns",
        "External PA capital (mm)": 500.0,
        "In-House annual return (%)": 4.0,  # Should NOT be converted as percentage here
        "risk_metrics": ["Return", "Risk", "ShortfallProb"],
    }

    config = load_config(config_data)
    assert config.N_SIMULATIONS == 100
    assert config.analysis_mode == "returns"
    assert config.external_pa_capital == 500.0
    assert config.mu_H == 4.0  # Raw value, no percentage conversion in YAML loading
