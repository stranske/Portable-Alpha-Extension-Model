"""Pydantic schemas for the wizard stepper configuration."""

from __future__ import annotations

from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field, model_validator


class AnalysisMode(str, Enum):
    """Analysis mode options for parameter sweep."""
    CAPITAL = "capital"
    RETURNS = "returns" 
    ALPHA_SHARES = "alpha_shares"
    VOL_MULT = "vol_mult"


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
    
    # Step 2: Capital Allocation (in millions)
    total_fund_capital: float = Field(default=300.0, gt=0, description="Total fund capital ($MM)")
    external_pa_capital: float = Field(default=100.0, ge=0, description="External portable alpha capital ($MM)")
    active_ext_capital: float = Field(default=50.0, ge=0, description="Active extension capital ($MM)")
    internal_pa_capital: float = Field(default=150.0, ge=0, description="Internal portable alpha capital ($MM)")
    
    # Portfolio weights and shares
    w_beta_h: float = Field(default=0.5, ge=0, le=1, description="Internal sleeve beta weight")
    w_alpha_h: float = Field(default=0.5, ge=0, le=1, description="Internal sleeve alpha weight")
    theta_extpa: float = Field(default=0.5, ge=0, le=1, description="External PA alpha fraction")
    active_share: float = Field(default=0.5, ge=0, le=1, description="Active-extension active share fraction")
    
    # Step 3: Return & Risk Parameters (annual)
    mu_h: float = Field(default=0.04, description="Annual mean return of in-house alpha")
    sigma_h: float = Field(default=0.01, gt=0, description="Annual volatility of in-house alpha") 
    mu_e: float = Field(default=0.05, description="Annual mean return of extension alpha")
    sigma_e: float = Field(default=0.02, gt=0, description="Annual volatility of extension alpha")
    mu_m: float = Field(default=0.03, description="Annual mean return of external PA alpha")
    sigma_m: float = Field(default=0.02, gt=0, description="Annual volatility of external PA alpha")
    
    # Step 4: Correlation Parameters
    rho_idx_h: float = Field(default=0.05, ge=-0.99, le=0.99, description="Correlation index vs in-house")
    rho_idx_e: float = Field(default=0.00, ge=-0.99, le=0.99, description="Correlation index vs extension")  
    rho_idx_m: float = Field(default=0.00, ge=-0.99, le=0.99, description="Correlation index vs external PA")
    rho_h_e: float = Field(default=0.10, ge=-0.99, le=0.99, description="Correlation in-house vs extension")
    rho_h_m: float = Field(default=0.10, ge=-0.99, le=0.99, description="Correlation in-house vs external PA")
    rho_e_m: float = Field(default=0.00, ge=-0.99, le=0.99, description="Correlation extension vs external PA")
    
    # Financing Parameters (monthly)
    internal_financing_mean_month: float = Field(default=0.0, description="Internal financing mean per month")
    internal_financing_sigma_month: float = Field(default=0.0, ge=0, description="Internal financing volatility per month")
    internal_spike_prob: float = Field(default=0.0, ge=0, le=1, description="Internal financing spike probability")
    internal_spike_factor: float = Field(default=0.0, ge=0, description="Internal financing spike size multiplier")
    
    ext_pa_financing_mean_month: float = Field(default=0.0, description="External PA financing mean per month")
    ext_pa_financing_sigma_month: float = Field(default=0.0, ge=0, description="External PA financing volatility per month") 
    ext_pa_spike_prob: float = Field(default=0.0, ge=0, le=1, description="External PA financing spike probability")
    ext_pa_spike_factor: float = Field(default=0.0, ge=0, description="External PA financing spike size multiplier")
    
    act_ext_financing_mean_month: float = Field(default=0.0, description="Active extension financing mean per month")
    act_ext_financing_sigma_month: float = Field(default=0.0, ge=0, description="Active extension financing volatility per month")
    act_ext_spike_prob: float = Field(default=0.0, ge=0, le=1, description="Active extension financing spike probability")
    act_ext_spike_factor: float = Field(default=0.0, ge=0, description="Active extension financing spike size multiplier")
    
    # Risk Metrics
    risk_metrics: List[RiskMetric] = Field(
        default=[RiskMetric.RETURN, RiskMetric.RISK, RiskMetric.SHORTFALL_PROB],
        description="Risk metrics to calculate"
    )
    
    @model_validator(mode="after")
    def validate_capital_allocation(self) -> "WizardScenarioConfig":
        """Validate capital allocation sums correctly."""
        allocated = self.external_pa_capital + self.active_ext_capital + self.internal_pa_capital
        if abs(allocated - self.total_fund_capital) > 0.01:  # Allow small rounding errors
            raise ValueError(f"Capital allocation ({allocated:.2f}) must equal total fund capital ({self.total_fund_capital:.2f})")
        return self
    
    @model_validator(mode="after") 
    def validate_weights(self) -> "WizardScenarioConfig":
        """Validate weight parameters sum to 1."""
        if abs(self.w_alpha_h + self.w_beta_h - 1.0) > 0.01:
            raise ValueError("w_alpha_h + w_beta_h must equal 1.0")
        return self
    
    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with existing YAML configs."""
        # Use model_dump with mode='json' to properly serialize enums
        data = self.model_dump(mode='json')
        
        # Convert field names to match existing YAML format
        yaml_data = {
            "N_SIMULATIONS": data["n_simulations"],
            "N_MONTHS": data["n_months"],
            "analysis_mode": data["analysis_mode"],
            "total_fund_capital": data["total_fund_capital"],
            "external_pa_capital": data["external_pa_capital"],
            "active_ext_capital": data["active_ext_capital"], 
            "internal_pa_capital": data["internal_pa_capital"],
            "w_beta_H": data["w_beta_h"],
            "w_alpha_H": data["w_alpha_h"],
            "theta_extpa": data["theta_extpa"],
            "active_share": data["active_share"],
            "mu_H": data["mu_h"],
            "sigma_H": data["sigma_h"],
            "mu_E": data["mu_e"],
            "sigma_E": data["sigma_e"],
            "mu_M": data["mu_m"],
            "sigma_M": data["sigma_m"],
            "rho_idx_H": data["rho_idx_h"],
            "rho_idx_E": data["rho_idx_e"],
            "rho_idx_M": data["rho_idx_m"],
            "rho_H_E": data["rho_h_e"],
            "rho_H_M": data["rho_h_m"],
            "rho_E_M": data["rho_e_m"],
            "internal_financing_mean_month": data["internal_financing_mean_month"],
            "internal_financing_sigma_month": data["internal_financing_sigma_month"],
            "internal_spike_prob": data["internal_spike_prob"],
            "internal_spike_factor": data["internal_spike_factor"],
            "ext_pa_financing_mean_month": data["ext_pa_financing_mean_month"],
            "ext_pa_financing_sigma_month": data["ext_pa_financing_sigma_month"],
            "ext_pa_spike_prob": data["ext_pa_spike_prob"],
            "ext_pa_spike_factor": data["ext_pa_spike_factor"],
            "act_ext_financing_mean_month": data["act_ext_financing_mean_month"],
            "act_ext_financing_sigma_month": data["act_ext_financing_sigma_month"],
            "act_ext_spike_prob": data["act_ext_spike_prob"],
            "act_ext_spike_factor": data["act_ext_spike_factor"],
            "risk_metrics": data["risk_metrics"],
        }
        
        return yaml_data


def get_default_config(analysis_mode: AnalysisMode) -> WizardScenarioConfig:
    """Get default configuration for the specified analysis mode."""
    base_config = WizardScenarioConfig(analysis_mode=analysis_mode)
    
    # Customize defaults based on analysis mode
    if analysis_mode == AnalysisMode.CAPITAL:
        # Focus on capital allocation optimization
        base_config.external_pa_capital = 0.0
        base_config.active_ext_capital = 0.0
        base_config.internal_pa_capital = 300.0
        
    elif analysis_mode == AnalysisMode.RETURNS:
        # Balanced allocation for return analysis
        base_config.external_pa_capital = 100.0
        base_config.active_ext_capital = 50.0
        base_config.internal_pa_capital = 150.0
        
    elif analysis_mode == AnalysisMode.ALPHA_SHARES:
        # Focus on alpha allocation
        base_config.theta_extpa = 0.3
        base_config.active_share = 0.3
        base_config.external_pa_capital = 150.0
        base_config.active_ext_capital = 75.0
        base_config.internal_pa_capital = 75.0
        
    elif analysis_mode == AnalysisMode.VOL_MULT:
        # Conservative allocation for volatility testing
        base_config.sigma_h = 0.005
        base_config.sigma_e = 0.01
        base_config.sigma_m = 0.01
        
    return base_config