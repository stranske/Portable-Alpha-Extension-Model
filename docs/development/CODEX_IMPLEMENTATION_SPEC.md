# Updated Implementation Requirements for Codex

## 🚨 IMPORTANT: CORE BUGS ALREADY FIXED

**DO NOT MODIFY THESE FILES** - They are working correctly:
- ✅ `pa_core/agents/active_ext.py` - Active share conversion fixed and tested
- ✅ `pa_core/cli.py` - Basic CLI functionality working  
- ✅ `dashboard/app.py` - Core dashboard features working
- ✅ All agent classes - Math and validation working correctly

**Focus on NEW functionality only.**

## MISSING FUNCTIONALITY TO IMPLEMENT

### 1. Parameter Sweep Engine (HIGH PRIORITY)

The current CLI only supports single simulations, but documentation promises 4 analysis modes with parameter sweeps.

#### Required Implementation:

**A. Analysis Mode Support in ModelConfig:**
```python
# Add to pa_core/config.py ModelConfig class:
analysis_mode: str = "returns"  # capital, returns, alpha_shares, vol_mult

# Add validation:
@model_validator(mode="after")
def check_analysis_mode(self) -> "ModelConfig":
    valid_modes = ["capital", "returns", "alpha_shares", "vol_mult"]
    if self.analysis_mode not in valid_modes:
        raise ValueError(f"analysis_mode must be one of: {valid_modes}")
    return self
```

**B. Parameter Sweep Parameters in ModelConfig:**
```python
# Capital mode sweep parameters:
max_external_combined_pct: float = 30.0
external_step_size_pct: float = 5.0

# Returns mode sweep parameters:  
in_house_return_min_pct: float = 2.0
in_house_return_max_pct: float = 6.0
in_house_return_step_pct: float = 2.0
in_house_vol_min_pct: float = 1.0
in_house_vol_max_pct: float = 3.0
in_house_vol_step_pct: float = 1.0
alpha_ext_return_min_pct: float = 1.0
alpha_ext_return_max_pct: float = 5.0
alpha_ext_return_step_pct: float = 2.0
alpha_ext_vol_min_pct: float = 2.0
alpha_ext_vol_max_pct: float = 4.0
alpha_ext_vol_step_pct: float = 1.0

# Alpha shares mode sweep parameters:
external_pa_alpha_min_pct: float = 25.0
external_pa_alpha_max_pct: float = 75.0
external_pa_alpha_step_pct: float = 5.0
active_share_min_pct: float = 20.0
active_share_max_pct: float = 100.0
active_share_step_pct: float = 5.0

# Vol mult mode sweep parameters:
sd_multiple_min: float = 2.0
sd_multiple_max: float = 4.0
sd_multiple_step: float = 0.25
```

**C. Sweep Engine Implementation:**
```python
# Create pa_core/sweep.py

from typing import List, Dict, Any, Iterator
from .config import ModelConfig
import numpy as np

def generate_parameter_combinations(cfg: ModelConfig) -> Iterator[Dict[str, Any]]:
    """Generate parameter combinations based on analysis_mode."""
    
    if cfg.analysis_mode == "capital":
        # Generate capital allocation combinations
        for ext_pct in np.arange(0, cfg.max_external_combined_pct + cfg.external_step_size_pct, cfg.external_step_size_pct):
            for act_pct in np.arange(0, ext_pct + cfg.external_step_size_pct, cfg.external_step_size_pct):
                ext_pa_pct = ext_pct - act_pct
                internal_pct = 100 - ext_pct
                
                yield {
                    "external_pa_capital": (ext_pa_pct / 100) * cfg.total_fund_capital,
                    "active_ext_capital": (act_pct / 100) * cfg.total_fund_capital,
                    "internal_pa_capital": (internal_pct / 100) * cfg.total_fund_capital,
                }
    
    elif cfg.analysis_mode == "returns":
        # Generate return assumption combinations
        for mu_H in np.arange(cfg.in_house_return_min_pct, cfg.in_house_return_max_pct + cfg.in_house_return_step_pct, cfg.in_house_return_step_pct):
            for sigma_H in np.arange(cfg.in_house_vol_min_pct, cfg.in_house_vol_max_pct + cfg.in_house_vol_step_pct, cfg.in_house_vol_step_pct):
                for mu_E in np.arange(cfg.alpha_ext_return_min_pct, cfg.alpha_ext_return_max_pct + cfg.alpha_ext_return_step_pct, cfg.alpha_ext_return_step_pct):
                    for sigma_E in np.arange(cfg.alpha_ext_vol_min_pct, cfg.alpha_ext_vol_max_pct + cfg.alpha_ext_vol_step_pct, cfg.alpha_ext_vol_step_pct):
                        yield {
                            "mu_H": mu_H / 100,
                            "sigma_H": sigma_H / 100, 
                            "mu_E": mu_E / 100,
                            "sigma_E": sigma_E / 100,
                        }
    
    elif cfg.analysis_mode == "alpha_shares":
        # Generate alpha share combinations
        for theta_extpa in np.arange(cfg.external_pa_alpha_min_pct, cfg.external_pa_alpha_max_pct + cfg.external_pa_alpha_step_pct, cfg.external_pa_alpha_step_pct):
            for active_share in np.arange(cfg.active_share_min_pct, cfg.active_share_max_pct + cfg.active_share_step_pct, cfg.active_share_step_pct):
                yield {
                    "theta_extpa": theta_extpa / 100,
                    "active_share": active_share,  # Keep as percentage for ActiveExtension agent
                }
    
    elif cfg.analysis_mode == "vol_mult":
        # Generate volatility multiplier combinations
        for sd_mult in np.arange(cfg.sd_multiple_min, cfg.sd_multiple_max + cfg.sd_multiple_step, cfg.sd_multiple_step):
            yield {
                "sigma_H": cfg.sigma_H * sd_mult,
                "sigma_E": cfg.sigma_E * sd_mult,
                "sigma_M": cfg.sigma_M * sd_mult,
            }

def run_parameter_sweep(cfg: ModelConfig, index_series, rng_returns, fin_rngs) -> List[Dict[str, Any]]:
    """Run parameter sweep and return results for all combinations."""
    results = []
    
    for i, param_overrides in enumerate(generate_parameter_combinations(cfg)):
        # Create modified config for this combination
        modified_cfg = cfg.model_copy(update=param_overrides)
        
        # Run single simulation with modified config
        # ... (implement simulation logic)
        
        # Store results with parameter combination info
        result = {
            "combination_id": i,
            "parameters": param_overrides,
            "summary": summary_table,  # Results from simulation
        }
        results.append(result)
    
    return results
```

### 2. CLI Enhancement for Analysis Modes

**Update pa_core/cli.py main function:**
```python
# Add after config loading:
if hasattr(cfg, 'analysis_mode') and cfg.analysis_mode != "returns":
    # Run parameter sweep
    from .sweep import run_parameter_sweep
    results = run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)
    
    # Export sweep results to Excel with multiple sheets
    from .reporting.excel import export_sweep_results
    export_sweep_results(results, filename=args.output)
else:
    # Run single simulation (current logic)
    # ... existing code ...
```

### 3. Template Standardization

**Create 4 mode-specific templates in config/:**

1. **config/capital_mode_template.csv** - Capital allocation sweeps
2. **config/returns_mode_template.csv** - Return assumption sweeps  
3. **config/alpha_shares_mode_template.csv** - Alpha share sweeps
4. **config/vol_mult_mode_template.csv** - Volatility multiplier sweeps

Each template should include:
- Clear parameter explanations for the specific analysis mode
- Appropriate sweep ranges and step sizes
- Business-relevant default values
- Mode-specific guidance in comments

### 4. Documentation Updates

**Update docs/UserGuide.md to:**
- Clearly explain what each analysis mode does
- Provide realistic parameter sweep examples
- Show how to interpret multi-scenario results
- Remove references to non-functional features

### 5. Validation and Testing

**Create comprehensive tests:**
- Each analysis mode produces expected parameter combinations
- Parameter sweeps generate reasonable result ranges
- ActiveExtension agent handles percentage conversion correctly
- All templates work with CLI without errors

## BUSINESS LOGIC REQUIREMENTS

### Capital Mode:
- **Purpose**: Test different capital allocation strategies
- **Key Parameters**: External PA %, Active Extension %, Internal PA %
- **Constraint**: All allocations must sum to 100%
- **Output Focus**: Risk-return efficiency by allocation

### Returns Mode:
- **Purpose**: Test sensitivity to return assumptions
- **Key Parameters**: Expected returns and volatilities for each sleeve
- **Constraint**: Returns should be realistic (2-6% alpha range)
- **Output Focus**: Impact of return assumptions on portfolio metrics

### Alpha Shares Mode:
- **Purpose**: Optimize alpha capture efficiency
- **Key Parameters**: External PA alpha fraction, Active share %
- **Constraint**: Alpha shares between 20-100%
- **Output Focus**: Alpha generation efficiency vs tracking error

### Vol Mult Mode:
- **Purpose**: Stress test under different volatility regimes
- **Key Parameters**: Volatility multipliers (2x, 3x, 4x baseline)
- **Constraint**: Multipliers should be reasonable (1.5x to 5x)
- **Output Focus**: Risk management under volatility stress

## IMPLEMENTATION PRIORITY

1. **HIGH**: Parameter sweep engine and analysis mode support
2. **HIGH**: Template standardization and validation
3. **MEDIUM**: Enhanced Excel output for sweep results
4. **MEDIUM**: Documentation updates
5. **LOW**: Advanced visualization for sweep results

This specification provides Codex with the complete requirements to implement the missing functionality while building on the critical bug fixes already completed.
