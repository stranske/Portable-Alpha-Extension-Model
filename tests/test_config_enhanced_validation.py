"""Integration tests for enhanced validation in config loading."""

from __future__ import annotations

import pytest

from pa_core.config import load_config


class TestEnhancedConfigValidation:
    """Test enhanced validation in ModelConfig."""

    def test_config_with_valid_params(self):
        """Test config loading with valid parameters passes all validations."""
        config_data = {
            "N_SIMULATIONS": 1000,
            "N_MONTHS": 12,
            "external_pa_capital": 200.0,
            "active_ext_capital": 300.0,
            "internal_pa_capital": 400.0,
            "total_fund_capital": 1000.0,
            "risk_metrics": ["Return", "Risk", "ShortfallProb"],
            "analysis_mode": "returns",
            # Add required step sizes
            "external_step_size_pct": 5.0,
            "in_house_return_step_pct": 2.0,
            "in_house_vol_step_pct": 1.0,
            "alpha_ext_return_step_pct": 2.0,
            "alpha_ext_vol_step_pct": 1.0,
            "external_pa_alpha_step_pct": 5.0,
            "active_share_step_pct": 5.0,
            "sd_multiple_step": 0.25,
        }

        # Should load successfully without raising errors
        config = load_config(config_data)
        assert config.N_SIMULATIONS == 1000
        assert config.external_pa_capital == 200.0

    def test_config_with_low_n_simulations_fails(self):
        """Test that very low N_SIMULATIONS raises validation error."""
        config_data = {
            "N_SIMULATIONS": 50,  # Too low
            "N_MONTHS": 12,
            "risk_metrics": ["Return", "Risk", "ShortfallProb"],
            "analysis_mode": "returns",
            # Add required step sizes
            "external_step_size_pct": 5.0,
            "in_house_return_step_pct": 2.0,
            "in_house_vol_step_pct": 1.0,
            "alpha_ext_return_step_pct": 2.0,
            "alpha_ext_vol_step_pct": 1.0,
            "external_pa_alpha_step_pct": 5.0,
            "active_share_step_pct": 5.0,
            "sd_multiple_step": 0.25,
        }

        with pytest.raises(ValueError, match="too low"):
            load_config(config_data)

    def test_config_with_invalid_step_size_fails(self):
        """Test that invalid step sizes raise validation errors."""
        config_data = {
            "N_SIMULATIONS": 1000,
            "N_MONTHS": 12,
            "risk_metrics": ["Return", "Risk", "ShortfallProb"],
            "analysis_mode": "returns",
            "external_step_size_pct": 0.0,  # Invalid - must be positive
            "in_house_return_step_pct": 2.0,
            "in_house_vol_step_pct": 1.0,
            "alpha_ext_return_step_pct": 2.0,
            "alpha_ext_vol_step_pct": 1.0,
            "external_pa_alpha_step_pct": 5.0,
            "active_share_step_pct": 5.0,
            "sd_multiple_step": 0.25,
        }

        with pytest.raises(ValueError, match="must be positive"):
            load_config(config_data)

    def test_config_with_capital_exceeding_margin_limit(self):
        """Test that capital allocation exceeding margin limits fails."""
        config_data = {
            "N_SIMULATIONS": 1000,
            "N_MONTHS": 12,
            "external_pa_capital": 100.0,
            "active_ext_capital": 100.0,
            "internal_pa_capital": 700.0,  # Total = 900 < 1000, but margin + internal > 1000
            "total_fund_capital": 1000.0,
            "risk_metrics": ["Return", "Risk", "ShortfallProb"],
            "analysis_mode": "returns",
            # Add margin fields - these would trigger the margin validation
            "reference_sigma": 0.05,  # 5% monthly vol
            "volatility_multiple": 8.0,  # High multiplier -> 400M margin requirement
            # Add required step sizes
            "external_step_size_pct": 5.0,
            "in_house_return_step_pct": 2.0,
            "in_house_vol_step_pct": 1.0,
            "alpha_ext_return_step_pct": 2.0,
            "alpha_ext_vol_step_pct": 1.0,
            "external_pa_alpha_step_pct": 5.0,
            "active_share_step_pct": 5.0,
            "sd_multiple_step": 0.25,
        }

        # Should fail due to margin + internal_pa exceeding total
        # margin = 0.05 * 8.0 * 1000 = 400M
        # internal_pa = 700M
        # total = 1100M > 1000M
        with pytest.raises(ValueError, match="Margin requirement.*exceeds total capital"):
            load_config(config_data)
