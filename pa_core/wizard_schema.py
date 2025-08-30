"""Enums and schemas for the portfolio wizard interface.

This module defines the configuration options and validation schemas
used by portfolio managers in the guided wizard interface.
"""

from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field, model_validator


# Analysis mode constants moved out of class for better maintainability
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
and validating risk management frameworks under stress conditions."""
}

ANALYSIS_MODE_DISPLAY_NAMES = {
    "capital": "Capital Allocation Analysis",
    "returns": "Return Assumption Sensitivity Analysis", 
    "alpha_shares": "Alpha Capture Efficiency Analysis",
    "vol_mult": "Volatility Stress Testing"
}


class AnalysisMode(str, Enum):
    """Analysis mode options for parameter sweep."""
    CAPITAL = "capital"
    RETURNS = "returns" 
    ALPHA_SHARES = "alpha_shares"
    VOL_MULT = "vol_mult"
    
    @property
    def description(self) -> str:
        """Get detailed description of the analysis mode for portfolio managers."""
        if self.value not in ANALYSIS_MODE_DESCRIPTIONS:
            raise ValueError(f"No description found for AnalysisMode value '{self.value}'. Please update the descriptions dictionary.")
        return ANALYSIS_MODE_DESCRIPTIONS[self.value]
    
    @property
    def display_name(self) -> str:
        """Get human-readable display name for the analysis mode."""
        return ANALYSIS_MODE_DISPLAY_NAMES.get(self.value, self.value.title())


class RiskMetric(str, Enum):
    """Risk metrics to calculate."""
    RETURN = "Return"
    RISK = "Risk"
    SHORTFALL_PROB = "ShortfallProb"


class WizardScenarioConfig(BaseModel):
    """Complete scenario configuration for wizard stepper."""
    
    # Step 1: Analysis Mode & Basic Settings
    analysis_mode: AnalysisMode = Field(default=AnalysisMode.RETURNS, description="Analysis mode for parameter sweep")
    n_simulations: int = Field(default=1000, ge=100, le=10000, description="Number of Monte Carlo trials")
    n_months: int = Field(default=12, ge=1, le=60, description="Months in each simulation run")

    """Analysis mode options for parameter sweep.

    
    Defines the primary focus and methodology for portfolio analysis,
    helping portfolio managers choose the right approach for their
    investment strategy evaluation needs.
    
    Each analysis mode provides specific guidance for portfolio managers:
    
    CAPITAL: Capital Allocation Analysis - Tests different capital allocation 
    strategies across portfolio sleeves. Use when determining the optimal mix
    of internal PA, external PA, and active extension allocations.
    
    RETURNS: Return Assumption Sensitivity Analysis - Tests sensitivity to 
    expected return assumptions while keeping capital allocation fixed. Ideal
    for validating return assumptions and understanding forecast impacts.
    
    ALPHA_SHARES: Alpha Capture Efficiency Analysis - Analyzes allocation of
    alpha sources across strategies to optimize alpha capture efficiency.
    Focuses on balancing alpha generation with tracking error.
    
    VOL_MULT: Volatility Stress Testing - Stress tests portfolio performance
    under different volatility regimes. Essential for risk management and
    extreme event preparation.
    """
    
    CAPITAL = "capital"
    RETURNS = "returns"
    ALPHA_SHARES = "alpha_shares"
    VOL_MULT = "vol_mult"
    
    @property
    def description(self) -> str:
        """Get detailed description of the analysis mode for portfolio managers."""
        descriptions = {
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
and validating risk management frameworks under stress conditions."""
        }
        if self.value not in descriptions:
            raise ValueError(f"No description found for AnalysisMode value '{self.value}'. Please update the descriptions dictionary.")
        return descriptions[self.value]
    
    @property
    def display_name(self) -> str:
        """Get human-readable display name for the analysis mode."""
        names = {
            "capital": "Capital Allocation Analysis",
            "returns": "Return Assumption Sensitivity Analysis", 
            "alpha_shares": "Alpha Capture Efficiency Analysis",
            "vol_mult": "Volatility Stress Testing"
        }
        return names.get(self.value, self.value.title())


class RiskMetric(str, Enum):
    """Risk metrics to calculate.
    
    Defines the key performance and risk metrics that will be
    calculated and reported for portfolio analysis.
    """
    
    RETURN = "Return"
    """Expected portfolio return metric."""
    
    RISK = "Risk" 
    """Portfolio risk/volatility metric."""
    
    SHORTFALL_PROB = "ShortfallProb"
    """Probability of failing to meet return targets."""
