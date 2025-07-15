# Tutorial 1 - Parameter Configuration Guide for Portable Alpha Practitioners

## Overview
This model simulates a **three-sleeve portable alpha strategy**:
1. **Internal PA** - Your in-house team manages both alpha and beta
2. **External PA** - External manager provides alpha, you manage beta separately  
3. **Active Extension** - Traditional active management overlay

## Parameter Explanation by Category

### 1. Simulation Settings
- **Number of simulations**: Monte Carlo trials (1000+ recommended for stable results)
- **Number of months**: Time horizon for each simulation (12 = 1 year)
- **Analysis mode**: Choose `returns`, `capital`, `alpha_shares` or `vol_mult` to specify a sweep or single-scenario run.

### 2. Capital Allocation (in millions)
Think of this as your asset allocation decision:
- **External PA capital**: Money allocated to external portable alpha managers
- **Active Extension capital**: Capital for active equity overlay strategies
- **Internal PA capital**: Assets managed internally for portable alpha
- **Total fund capital**: Must equal sum of the three sleeves above

**Example Business Logic**: 
- Large institutional fund ($300M) might allocate:
  - 50% ($150M) to internal PA (lower fees, full control)
  - 33% ($100M) to external PA (access to specialized alpha)
  - 17% ($50M) to active extension (tactical opportunities)

### 3. Internal PA Sleeve Configuration
- **In-House beta share**: What portion of internal capital tracks the benchmark (0.5 = 50%)
- **In-House alpha share**: What portion seeks active return (0.5 = 50%)
- **Note**: These should sum to 1.0 for full capital deployment

**Business Logic**: Higher beta share = more benchmark tracking, lower tracking error

### 4. External PA Manager Configuration  
- **External PA alpha fraction**: How much of external manager's mandate is pure alpha vs beta
- **Typical Range**: 0.3-0.7 (30-70% alpha focus)

### 5. Active Extension Configuration
- **Active share (%)**: How much the active manager deviates from benchmark
- **Typical Range**: 30-80% (higher = more active, higher tracking error)

### 6. Expected Return Assumptions (Annual %)
Set these based on your capital market assumptions:
- **In-House annual return**: Expected alpha from internal team (e.g., 2-6%)
- **Alpha-Extension annual return**: Expected alpha from active overlay (e.g., 3-7%)  
- **External annual return**: Expected alpha from external PA manager (e.g., 2-5%)

**Calibration Tip**: Use historical performance or market research for realistic assumptions

### 7. Risk Parameters (Annual %)
- **In-House annual vol**: Volatility of internal alpha generation (typically 1-3%)
- **Alpha-Extension annual vol**: Volatility of active extension returns (typically 2-5%)
- **External annual vol**: Volatility of external manager alpha (typically 1-4%)

**Risk Budgeting**: Higher vol = higher expected return but more uncertainty

### 8. Correlation Matrix
Critical for understanding diversification benefits:
- **Corr index–[Sleeve]**: How much each alpha source correlates with benchmark
  - Lower is better for diversification (typically -0.1 to 0.2)
- **Corr [Sleeve]–[Sleeve]**: Cross-correlations between alpha sources
  - Lower correlations improve risk-adjusted returns

**Portfolio Construction Insight**: Negative or low correlations enhance Sharpe ratio

### 9. Financing Costs (Monthly %)
Models the cost of leverage/shorting in portable alpha:
- **Internal financing mean (monthly %)**: Baseline borrowing cost for the internal sleeve
- **Internal financing vol (monthly %)**: Volatility of internal financing
- **Internal monthly spike prob**: Probability of a financing spike internally
- **Internal spike multiplier**: Size multiplier applied when a spike occurs
- **External PA financing mean (monthly %)**: Baseline cost for external PA
- **External PA financing vol (monthly %)**: Volatility of external PA financing
- **External PA monthly spike prob**: Probability of a spike in external PA
- **External PA spike multiplier**: Size multiplier for external PA spikes
- **Active Ext financing mean (monthly %)**: Baseline cost for active extension
- **Active Ext financing vol (monthly %)**: Volatility of active extension financing
- **Active Ext monthly spike prob**: Probability of a spike for active extension
- **Active Ext spike multiplier**: Size multiplier for active extension spikes

## Quick Start Configuration for First-Time Users

For your first run, consider these conservative assumptions:
- Keep default capital allocation (50%/33%/17% split)
- Use modest alpha expectations (2-4% annually)
- Set low correlations (0.0-0.1) for diversification
- Start with zero financing costs to understand base case
- Use 1000+ simulations for stable results

## Key Validation Checks Before Running
1. **Capital adds up**: Sum of three sleeves = Total fund capital
2. **Shares sum to 1.0**: In-house beta + alpha shares = 1.0
3. **Realistic assumptions**: Returns and volatilities match market experience
4. **Risk metrics included**: Must have "ShortfallProb" in risk_metrics

This parameter file will generate a portable alpha simulation showing how your three-sleeve strategy performs under Monte Carlo stress testing.
