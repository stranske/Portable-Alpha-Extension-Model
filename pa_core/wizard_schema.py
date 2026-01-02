"""Enums and schemas for the portfolio wizard interface.

This module defines the configuration options and helper utilities
used by portfolio managers in the guided wizard interface.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from .config import ModelConfig

# Module-level constants for names and descriptions
ANALYSIS_MODE_DESCRIPTIONS = {
    "capital": """Capital Allocation Analysis

Tests different capital allocation strategies across portfolio sleeves.
Use when determining the optimal mix of internal PA, external PA,
and active extension allocations.

Key Parameters:
• External PA allocation percentage
• Active Extension allocation percentage
• Internal PA allocation percentage

Business Focus:
• Risk-return efficiency by allocation
• Diversification benefits across strategies
• Capital deployment optimization

Constraint: All allocations must sum to 100% of total fund capital.

Ideal for: Portfolio managers evaluating strategic asset allocation
decisions and comparing different capital deployment strategies.""",
    "returns": """Return Assumption Sensitivity Analysis

Tests sensitivity to expected return assumptions while keeping
capital allocation fixed. Evaluates how changes in alpha expectations
impact portfolio performance metrics.

Key Parameters:
• Expected returns for each sleeve
• Volatility assumptions
• Risk-adjusted return forecasts

Business Focus:
• Impact of return assumptions on portfolio metrics
• Stress testing alpha forecasts
• Sensitivity to market outlook changes

Constraint: Returns should be realistic (typically 2-6% alpha range).

Ideal for: Portfolio managers validating return assumptions and
understanding how forecast changes affect portfolio outcomes.""",
    "alpha_shares": """Alpha Capture Efficiency Analysis

Analyzes the allocation of alpha sources across different strategies
to optimize alpha capture efficiency. Focuses on the balance between
alpha generation and tracking error.

Key Parameters:
• External PA alpha fraction
• Active share percentages
• Alpha/beta split ratios

Business Focus:
• Alpha generation efficiency vs tracking error
• Optimal balance between active and passive exposure
• Risk-adjusted alpha capture

Constraint: Alpha shares typically range between 20-100%.

Ideal for: Portfolio managers optimizing the balance between
alpha generation and risk control, particularly in portable
alpha strategies.""",
    "vol_mult": """Volatility Stress Testing

Stress tests portfolio performance under different volatility
regimes to evaluate risk management effectiveness. Essential
for understanding portfolio behavior in extreme market conditions.

Key Parameters:
• Volatility multipliers (2x, 3x, 4x baseline)
• Stress scenario definitions
• Risk factor scaling

Business Focus:
• Risk management under volatility stress
• Portfolio resilience in extreme conditions
• Downside protection evaluation

Constraint: Multipliers should be reasonable (typically 1.5x to 5x).

Ideal for: Portfolio managers preparing for extreme market events
and validating risk management frameworks under stress conditions.""",
}

ANALYSIS_MODE_DISPLAY_NAMES = {
    "capital": "Capital Allocation Analysis",
    "returns": "Return Assumption Sensitivity Analysis",
    "alpha_shares": "Alpha Capture Efficiency Analysis",
    "vol_mult": "Volatility Stress Testing",
}


class AnalysisMode(str, Enum):
    """Analysis mode options for parameter sweep.

    Designed for portfolio managers using the wizard to explore
    capital, returns, alpha shares, and volatility scenarios.
    """

    CAPITAL = "capital"
    RETURNS = "returns"
    ALPHA_SHARES = "alpha_shares"
    VOL_MULT = "vol_mult"

    @property
    def description(self) -> str:
        """Return a detailed description of this analysis mode."""

        if self.value not in ANALYSIS_MODE_DESCRIPTIONS:
            raise ValueError(
                f"No description found for AnalysisMode value '{self.value}'. Please update the descriptions dictionary."
            )
        return ANALYSIS_MODE_DESCRIPTIONS[self.value]

    @property
    def display_name(self) -> str:
        """Return a human-readable display name for this analysis mode."""

        return ANALYSIS_MODE_DISPLAY_NAMES.get(self.value, self.value.title())


class RiskMetric(str, Enum):
    """Risk metrics to calculate."""

    RETURN = "Return"
    RISK = "Risk"
    SHORTFALL_PROB = "terminal_ShortfallProb"


class WizardScenarioConfig(BaseModel):  # Minimal placeholder for UI wiring
    """Minimal configuration shell used by the wizard UI.

    Tests don't depend on this class' internals; it's provided to
    keep imports stable and allow progressive enhancement.
    """

    analysis_mode: AnalysisMode = Field(
        default=AnalysisMode.RETURNS, description="Analysis mode for parameter sweep"
    )
    n_simulations: int = Field(
        default=1000, ge=100, le=10000, description="Number of Monte Carlo trials"
    )
    n_months: int = Field(default=12, ge=1, le=60, description="Months in each simulation run")


@dataclass
class DefaultConfigView:
    # Core simulation parameters
    analysis_mode: AnalysisMode
    n_simulations: int
    n_months: int

    # Capital allocation
    external_pa_capital: float
    active_ext_capital: float
    internal_pa_capital: float
    total_fund_capital: float

    # Portfolio shares and fractions
    w_beta_h: float
    w_alpha_h: float
    theta_extpa: float
    active_share: float

    # Expected returns
    mu_h: float
    mu_e: float
    mu_m: float

    # Volatilities
    sigma_h: float
    sigma_e: float
    sigma_m: float

    # Correlations
    rho_idx_h: float
    rho_idx_e: float
    rho_idx_m: float
    rho_h_e: float
    rho_h_m: float
    rho_e_m: float

    # Risk metrics
    risk_metrics: List[str]


def _make_view(m: ModelConfig) -> DefaultConfigView:
    """Create DefaultConfigView from ModelConfig with consistent field mappings.

    This function maps ModelConfig fields to DefaultConfigView fields, handling
    field name differences (e.g., N_SIMULATIONS -> n_simulations, mu_H -> mu_h).
    All defaults come from the validated ModelConfig instance.

    Args:
        m: Validated ModelConfig instance with all required defaults

    Returns:
        DefaultConfigView with all attributes populated from ModelConfig
    """
    return DefaultConfigView(
        # Core simulation parameters
        analysis_mode=AnalysisMode(m.analysis_mode),
        n_simulations=m.N_SIMULATIONS,
        n_months=m.N_MONTHS,
        # Capital allocation
        external_pa_capital=m.external_pa_capital,
        active_ext_capital=m.active_ext_capital,
        internal_pa_capital=m.internal_pa_capital,
        total_fund_capital=m.total_fund_capital,
        # Portfolio shares and fractions
        w_beta_h=m.w_beta_H,
        w_alpha_h=m.w_alpha_H,
        theta_extpa=m.theta_extpa,
        active_share=m.active_share,
        # Expected returns
        mu_h=m.mu_H,
        mu_e=m.mu_E,
        mu_m=m.mu_M,
        # Volatilities
        sigma_h=m.sigma_H,
        sigma_e=m.sigma_E,
        sigma_m=m.sigma_M,
        # Correlations
        rho_idx_h=m.rho_idx_H,
        rho_idx_e=m.rho_idx_E,
        rho_idx_m=m.rho_idx_M,
        rho_h_e=m.rho_H_E,
        rho_h_m=m.rho_H_M,
        rho_e_m=m.rho_E_M,
        # Risk metrics
        risk_metrics=m.risk_metrics,
    )


def get_default_config(mode: AnalysisMode) -> DefaultConfigView:
    """Return a mode-specific default config view used by tests and UI.

    The values are derived from the validated ModelConfig defaults and then
    gently adjusted per mode to satisfy high-level expectations.
    """

    # Build ModelConfig using alias names via model_validate to satisfy typing
    base = ModelConfig.model_validate({"Number of simulations": 1, "Number of months": 1})
    cfg = _make_view(base)

    if mode == AnalysisMode.CAPITAL:
        # Start with a non-zero internal allocation for capital sweep
        cfg.internal_pa_capital = max(cfg.internal_pa_capital, 1.0)

    elif mode == AnalysisMode.RETURNS:
        # Ensure balanced, non-zero sleeves for return analysis
        if cfg.external_pa_capital == 0:
            cfg.external_pa_capital = 1.0
        if cfg.active_ext_capital == 0:
            cfg.active_ext_capital = 1.0
        if cfg.internal_pa_capital == 0:
            cfg.internal_pa_capital = 1.0

    elif mode == AnalysisMode.ALPHA_SHARES:
        # Keep alpha and active share inside (0, 1)
        if not (0 < cfg.theta_extpa < 1):
            cfg.theta_extpa = 0.6
        if not (0 < cfg.active_share < 1):
            cfg.active_share = 0.6

    elif mode == AnalysisMode.VOL_MULT:
        # Slightly conservative vols vs. returns baseline
        returns_defaults = _make_view(
            ModelConfig.model_validate({"Number of simulations": 1, "Number of months": 1})
        )
        cfg.sigma_h = returns_defaults.sigma_h * 0.9
        cfg.sigma_e = returns_defaults.sigma_e * 0.9
        cfg.sigma_m = returns_defaults.sigma_m * 0.9

    return cfg
