"""Tests for the wizard schema and stepper functionality."""

import pytest
import yaml
import tempfile
from pathlib import Path

from pa_core.wizard_schema import (
    WizardScenarioConfig, 
    AnalysisMode, 
    get_default_config
)


class TestWizardScenarioConfig:
    """Test the wizard scenario configuration schema."""

    def test_default_config_creation(self):
        """Test that default configs can be created for all analysis modes."""
        for mode in AnalysisMode:
            config = get_default_config(mode)
            assert config.analysis_mode == mode
            assert config.n_simulations == 1000
            assert config.n_months == 12
            
    def test_config_validation_passes(self):
        """Test that default configs pass validation."""
        for mode in AnalysisMode:
            config = get_default_config(mode)
            # Should not raise any validation errors
            validated = WizardScenarioConfig.model_validate(config.model_dump())
            assert validated.analysis_mode == mode

    def test_capital_allocation_validation(self):
        """Test capital allocation validation."""
        config = get_default_config(AnalysisMode.RETURNS)
        
        # Valid allocation should pass
        config.total_fund_capital = 300.0
        config.external_pa_capital = 100.0
        config.active_ext_capital = 50.0
        config.internal_pa_capital = 150.0
        
        # Should not raise validation error
        WizardScenarioConfig.model_validate(config.model_dump())
        
        # Invalid allocation should fail
        config.internal_pa_capital = 200.0  # Now totals 350, not 300
        
        with pytest.raises(ValueError, match="Capital allocation.*must equal total fund capital"):
            WizardScenarioConfig.model_validate(config.model_dump())

    def test_weight_validation(self):
        """Test weight parameter validation."""
        config = get_default_config(AnalysisMode.RETURNS)
        
        # Valid weights should pass
        config.w_beta_h = 0.6
        config.w_alpha_h = 0.4
        WizardScenarioConfig.model_validate(config.model_dump())
        
        # Invalid weights should fail
        config.w_alpha_h = 0.5  # Now totals 1.1, not 1.0
        with pytest.raises(ValueError, match="w_alpha_h \\+ w_beta_h must equal 1.0"):
            WizardScenarioConfig.model_validate(config.model_dump())

    def test_correlation_bounds(self):
        """Test correlation parameter bounds."""
        config = get_default_config(AnalysisMode.RETURNS)
        
        # Valid correlations should pass
        config.rho_idx_h = 0.5
        config.rho_h_e = -0.3
        WizardScenarioConfig.model_validate(config.model_dump())
        
        # Invalid correlations should fail
        config.rho_idx_h = 1.5  # Outside [-0.99, 0.99]
        with pytest.raises(ValueError):
            WizardScenarioConfig.model_validate(config.model_dump())

    def test_yaml_dict_conversion(self):
        """Test conversion to YAML-compatible dictionary."""
        config = get_default_config(AnalysisMode.RETURNS)
        yaml_dict = config.to_yaml_dict()
        
        # Check required fields are present
        required_fields = [
            'N_SIMULATIONS', 'N_MONTHS', 'analysis_mode', 
            'total_fund_capital', 'external_pa_capital',
            'mu_H', 'sigma_H', 'rho_idx_H'
        ]
        
        for field in required_fields:
            assert field in yaml_dict, f"Missing required field: {field}"
        
        # Check types are serializable
        assert isinstance(yaml_dict['analysis_mode'], str)
        assert yaml_dict['analysis_mode'] == 'returns'
        assert isinstance(yaml_dict['N_SIMULATIONS'], int)
        assert isinstance(yaml_dict['risk_metrics'], list)

    def test_yaml_serialization(self):
        """Test that config can be serialized to valid YAML."""
        config = get_default_config(AnalysisMode.CAPITAL)
        yaml_dict = config.to_yaml_dict()
        
        # Should not raise any YAML serialization errors
        yaml_str = yaml.safe_dump(yaml_dict, default_flow_style=False)
        assert len(yaml_str) > 0
        
        # Should be able to load it back
        loaded = yaml.safe_load(yaml_str)
        assert loaded['analysis_mode'] == 'capital'
        assert loaded['N_SIMULATIONS'] == 1000

    def test_mode_specific_defaults(self):
        """Test that different modes have appropriate defaults."""
        capital_config = get_default_config(AnalysisMode.CAPITAL)
        returns_config = get_default_config(AnalysisMode.RETURNS)
        
        # Capital mode should have different allocations than returns mode
        assert capital_config.external_pa_capital != returns_config.external_pa_capital
        
        # Vol mult mode should have lower volatilities
        vol_config = get_default_config(AnalysisMode.VOL_MULT)
        assert vol_config.sigma_h < returns_config.sigma_h

    def test_config_file_compatibility(self):
        """Test that wizard configs are compatible with existing CLI."""
        config = get_default_config(AnalysisMode.ALPHA_SHARES)
        yaml_dict = config.to_yaml_dict()
        yaml_str = yaml.safe_dump(yaml_dict, default_flow_style=False)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_str)
            temp_path = f.name
        
        try:
            # Test that existing config loader can load it
            from pa_core.config import load_config
            loaded_config = load_config(temp_path)
            
            # Should have loaded successfully (returns ModelConfig object)
            assert hasattr(loaded_config, 'N_SIMULATIONS')
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


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