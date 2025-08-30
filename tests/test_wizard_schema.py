"""Test module for wizard schema enums."""

import pytest
from pa_core.wizard_schema import AnalysisMode, RiskMetric


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


class TestEnumIntegration:
    """Test integration between enums and expected usage patterns."""
    
    def test_analysis_mode_dashboard_compatibility(self):
        """Test that enums work with expected dashboard patterns."""
        # Test creating mode options mapping like in dashboard
        mode_options = {
            AnalysisMode.RETURNS: f"📈 {AnalysisMode.RETURNS.display_name}",
            AnalysisMode.CAPITAL: f"💰 {AnalysisMode.CAPITAL.display_name}",
            AnalysisMode.ALPHA_SHARES: f"🎯 {AnalysisMode.ALPHA_SHARES.display_name}",
            AnalysisMode.VOL_MULT: f"📊 {AnalysisMode.VOL_MULT.display_name}",
        }
        
        assert len(mode_options) == 4
        for mode, display_text in mode_options.items():
            assert isinstance(mode, AnalysisMode)
            assert isinstance(display_text, str)
            assert len(display_text) > 5  # Has emoji and text
    
    def test_string_enum_compatibility(self):
        """Test that enums work as strings for config compatibility."""
        # Test that enum values work as strings (AnalysisMode inherits from str)
        assert AnalysisMode.CAPITAL == "capital"
        assert AnalysisMode.RETURNS == "returns"
        assert AnalysisMode.ALPHA_SHARES == "alpha_shares"
        assert AnalysisMode.VOL_MULT == "vol_mult"
        
        # Test that enum values can be used in string contexts
        valid_modes = ["capital", "returns", "alpha_shares", "vol_mult"]
        for mode in AnalysisMode:
            assert mode.value in valid_modes
            assert mode in valid_modes  # Since it inherits from str