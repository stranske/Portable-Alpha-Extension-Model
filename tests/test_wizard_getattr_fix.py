"""Test that wizard schema has all required attributes, eliminating need for getattr() with defaults."""

from pa_core.config import ModelConfig
from pa_core.wizard_schema import AnalysisMode, get_default_config


class TestWizardConfigConsistency:
    """Test that DefaultConfigView has all attributes needed by dashboard without getattr() fallbacks."""

    def test_default_config_view_has_all_required_attributes(self):
        """Test that DefaultConfigView contains all attributes accessed in Scenario_Wizard."""
        # These are all the attributes accessed via getattr() in the original code
        required_attributes = [
            "analysis_mode",
            "n_simulations",
            "n_months",
            "total_fund_capital",
            "external_pa_capital",
            "active_ext_capital",
            "internal_pa_capital",
            "w_beta_h",
            "w_alpha_h",
            "theta_extpa",
            "active_share",
            "mu_h",
            "mu_e",
            "mu_m",
            "sigma_h",
            "sigma_e",
            "sigma_m",
            "rho_idx_h",
            "rho_idx_e",
            "rho_idx_m",
            "rho_h_e",
            "rho_h_m",
            "rho_e_m",
            "risk_metrics",
        ]

        # Test for all analysis modes
        for mode in [
            AnalysisMode.RETURNS,
            AnalysisMode.CAPITAL,
            AnalysisMode.ALPHA_SHARES,
            AnalysisMode.VOL_MULT,
        ]:
            config = get_default_config(mode)

            for attr in required_attributes:
                assert hasattr(
                    config, attr
                ), f"DefaultConfigView missing required attribute '{attr}' for mode {mode.value}"
                # Ensure attribute is not None and has a reasonable value
                value = getattr(config, attr)
                assert (
                    value is not None
                ), f"Attribute '{attr}' is None for mode {mode.value}"

    def test_default_config_values_match_model_config_defaults(self):
        """Test that DefaultConfigView values are consistent with ModelConfig defaults where applicable."""
        # Create a base ModelConfig with minimal required fields
        model_config = ModelConfig.model_validate(
            {"Number of simulations": 1000, "Number of months": 12}
        )

        # Get equivalent DefaultConfigView
        default_view = get_default_config(AnalysisMode.RETURNS)

        # Test that default values from ModelConfig are preserved in DefaultConfigView
        # (except where intentionally modified for specific analysis modes)
        assert default_view.total_fund_capital == model_config.total_fund_capital
        assert default_view.w_beta_h == model_config.w_beta_H
        assert default_view.w_alpha_h == model_config.w_alpha_H
        assert default_view.theta_extpa == model_config.theta_extpa
        assert default_view.active_share == model_config.active_share
        assert default_view.mu_h == model_config.mu_H
        assert default_view.mu_e == model_config.mu_E
        assert default_view.mu_m == model_config.mu_M
        assert default_view.sigma_h == model_config.sigma_H
        assert default_view.sigma_e == model_config.sigma_E
        assert default_view.sigma_m == model_config.sigma_M
        assert default_view.risk_metrics == model_config.risk_metrics

    def test_analysis_mode_specific_overrides(self):
        """Test that mode-specific adjustments work correctly."""

        # Test CAPITAL mode ensures internal_pa_capital >= 1.0
        capital_config = get_default_config(AnalysisMode.CAPITAL)
        assert capital_config.internal_pa_capital >= 1.0

        # Test RETURNS mode ensures all capital allocations are non-zero
        returns_config = get_default_config(AnalysisMode.RETURNS)
        assert returns_config.external_pa_capital > 0
        assert returns_config.active_ext_capital > 0
        assert returns_config.internal_pa_capital > 0

        # Test ALPHA_SHARES mode keeps shares in (0, 1) range
        alpha_config = get_default_config(AnalysisMode.ALPHA_SHARES)
        assert 0 < alpha_config.theta_extpa < 1
        assert 0 < alpha_config.active_share < 1

        # Test VOL_MULT mode has conservative volatilities
        vol_config = get_default_config(AnalysisMode.VOL_MULT)
        # Should be 90% of base model volatilities
        base_model = ModelConfig.model_validate(
            {"Number of simulations": 1, "Number of months": 1}
        )
        assert abs(vol_config.sigma_h - base_model.sigma_H * 0.9) < 1e-10
        assert abs(vol_config.sigma_e - base_model.sigma_E * 0.9) < 1e-10
        assert abs(vol_config.sigma_m - base_model.sigma_M * 0.9) < 1e-10

    def test_no_getattr_needed(self):
        """Test that all attributes can be accessed directly without getattr() fallbacks."""
        config = get_default_config(AnalysisMode.RETURNS)

        # These should all work without getattr()
        assert isinstance(config.analysis_mode, AnalysisMode)
        assert isinstance(config.n_simulations, int)
        assert isinstance(config.n_months, int)
        assert isinstance(config.total_fund_capital, float)
        assert isinstance(config.external_pa_capital, float)
        assert isinstance(config.active_ext_capital, float)
        assert isinstance(config.internal_pa_capital, float)
        assert isinstance(config.w_beta_h, float)
        assert isinstance(config.w_alpha_h, float)
        assert isinstance(config.theta_extpa, float)
        assert isinstance(config.active_share, float)
        assert isinstance(config.mu_h, float)
        assert isinstance(config.mu_e, float)
        assert isinstance(config.mu_m, float)
        assert isinstance(config.sigma_h, float)
        assert isinstance(config.sigma_e, float)
        assert isinstance(config.sigma_m, float)
        assert isinstance(config.rho_idx_h, float)
        assert isinstance(config.rho_idx_e, float)
        assert isinstance(config.rho_idx_m, float)
        assert isinstance(config.rho_h_e, float)
        assert isinstance(config.rho_h_m, float)
        assert isinstance(config.rho_e_m, float)
        assert isinstance(config.risk_metrics, list)
