"""Test module for wizard schema enums."""

import pytest
from pa_core.wizard_schema import AnalysisMode, RiskMetric

from pa_core.wizard_schema import (
    WizardScenarioConfig, 
    AnalysisMode, 
    RiskMetric,
    get_default_config,
    ANALYSIS_MODE_DESCRIPTIONS,
    ANALYSIS_MODE_DISPLAY_NAMES
)

class TestAnalysisMode:
    """Test AnalysisMode enum functionality."""
    
    def test_analysis_mode_values(self):
        """Test that all expected analysis mode values exist."""
        expected_values = ["capital", "returns", "alpha_shares", "vol_mult"]
        actual_values = [mode.value for mode in AnalysisMode]
        assert set(actual_values) == set(expected_values)
    
    def test_analysis_mode_descriptions(self):
        """Test that all analysis modes have descriptions."""
        for mode in AnalysisMode:
            description = mode.description
            assert isinstance(description, str)
            assert len(description) > 50  # Ensure substantial description
            assert "Key Parameters:" in description
            assert "Business Focus:" in description
            assert "Ideal for:" in description
    
    def test_analysis_mode_display_names(self):
        """Test that all analysis modes have proper display names."""
        expected_names = {
            "capital": "Capital Allocation Analysis",
            "returns": "Return Assumption Sensitivity Analysis", 
            "alpha_shares": "Alpha Capture Efficiency Analysis",
            "vol_mult": "Volatility Stress Testing"
        }
        
        for mode in AnalysisMode:
            assert mode.display_name == expected_names[mode.value]
    
    def test_analysis_mode_class_docstring(self):
        """Test that the enum class has proper documentation."""
        assert AnalysisMode.__doc__ is not None
        assert "parameter sweep" in AnalysisMode.__doc__.lower()
        assert "portfolio managers" in AnalysisMode.__doc__.lower()


class TestRiskMetric:
    """Test RiskMetric enum functionality."""
    
    def test_risk_metric_values(self):
        """Test that all expected risk metric values exist."""
        expected_values = ["Return", "Risk", "ShortfallProb"]
        actual_values = [metric.value for metric in RiskMetric]
        assert set(actual_values) == set(expected_values)
    
    def test_risk_metric_class_docstring(self):
        """Test that the enum class has proper documentation."""
        assert RiskMetric.__doc__ is not None
        assert "risk metrics" in RiskMetric.__doc__.lower()


class TestAnalysisModeDefaults:
    """Test analysis mode specific default configurations."""

    def test_capital_mode_defaults(self):
        """Test capital mode has appropriate defaults for capital optimization."""
        config = get_default_config(AnalysisMode.CAPITAL)
        # Should start with internal-only allocation for capital sweep
        assert config.internal_pa_capital > 0
        
    def test_returns_mode_defaults(self):
        """Test returns mode has balanced allocation for return analysis.""" 
        config = get_default_config(AnalysisMode.RETURNS)
        # Should have balanced allocation across all sleeves
        assert config.external_pa_capital > 0
        assert config.active_ext_capital > 0 
        assert config.internal_pa_capital > 0
        
    def test_alpha_shares_mode_defaults(self):
        """Test alpha shares mode optimizes alpha allocation."""
        config = get_default_config(AnalysisMode.ALPHA_SHARES)
        # Should have different alpha fractions for testing
        assert 0 < config.theta_extpa < 1
        assert 0 < config.active_share < 1
        
    def test_vol_mult_mode_defaults(self):
        """Test vol mult mode has conservative volatility settings."""
        config = get_default_config(AnalysisMode.VOL_MULT)
        returns_config = get_default_config(AnalysisMode.RETURNS)
        
        # Should have lower volatilities for stress testing
        assert config.sigma_h <= returns_config.sigma_h
        assert config.sigma_e <= returns_config.sigma_e
        assert config.sigma_m <= returns_config.sigma_m


class TestAnalysisModeProperties:
    """Test the new property methods for AnalysisMode enum."""
    
    def test_description_property_exists(self):
        """Test that all analysis modes have description property."""
        for mode in AnalysisMode:
            description = mode.description
            assert isinstance(description, str)
            assert len(description) > 50, f"Description for {mode.value} should be substantial"
    
    def test_display_name_property_exists(self):
        """Test that all analysis modes have display_name property."""
        for mode in AnalysisMode:
            display_name = mode.display_name
            assert isinstance(display_name, str)
            assert len(display_name) > 5, f"Display name for {mode.value} should be meaningful"
    
    def test_specific_display_names(self):
        """Test specific expected display names."""
        expected = {
            AnalysisMode.CAPITAL: "Capital Allocation Analysis",
            AnalysisMode.RETURNS: "Return Assumption Sensitivity Analysis",
            AnalysisMode.ALPHA_SHARES: "Alpha Capture Efficiency Analysis",
            AnalysisMode.VOL_MULT: "Volatility Stress Testing"
        }
        
        for mode, expected_name in expected.items():
            assert mode.display_name == expected_name
    
    def test_descriptions_contain_expected_content(self):
        """Test that descriptions contain expected analysis-specific content."""
        # Each description should contain the mode name
        assert "Capital Allocation Analysis" in AnalysisMode.CAPITAL.description
        assert "Return Assumption Sensitivity Analysis" in AnalysisMode.RETURNS.description
        assert "Alpha Capture Efficiency Analysis" in AnalysisMode.ALPHA_SHARES.description
        assert "Volatility Stress Testing" in AnalysisMode.VOL_MULT.description
        
        # Each should contain business-focused content
        assert "Business Focus:" in AnalysisMode.CAPITAL.description
        assert "Key Parameters:" in AnalysisMode.RETURNS.description
        assert "Ideal for:" in AnalysisMode.ALPHA_SHARES.description
        assert "Constraint:" in AnalysisMode.VOL_MULT.description
    
    def test_constants_are_comprehensive(self):
        """Test that constants cover all enum values."""
        enum_values = {mode.value for mode in AnalysisMode}
        
        # All enum values should have descriptions
        assert set(ANALYSIS_MODE_DESCRIPTIONS.keys()) == enum_values
        
        # All enum values should have display names
        assert set(ANALYSIS_MODE_DISPLAY_NAMES.keys()) == enum_values
    
    def test_constants_are_used_by_properties(self):
        """Test that the property methods use the module-level constants."""
        # This ensures the refactoring correctly moved constants out of embedded dictionaries
        for mode in AnalysisMode:
            # Description should come from the constant
            assert mode.description == ANALYSIS_MODE_DESCRIPTIONS[mode.value]
            
            # Display name should come from the constant
            assert mode.display_name == ANALYSIS_MODE_DISPLAY_NAMES[mode.value]
    
    def test_description_error_handling(self):
        """Test that description property handles missing values appropriately."""
        # Create a mock enum value that's not in the constants
        # This tests the error path without modifying the actual constants
        from unittest.mock import Mock
        
        mock_mode = Mock()
        mock_mode.value = "nonexistent_mode"
        
        # Bind the property method to our mock
        description_method = AnalysisMode.description.fget
        
        with pytest.raises(ValueError, match="No description found for AnalysisMode value 'nonexistent_mode'"):
            description_method(mock_mode)
