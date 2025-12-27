# Tutorial 1: Multi-Mode Parameter Sweep Demonstration 

**🎯 Goal**: Transform basic single-scenario analysis into professional multi-scenario analysis workflows using the Parameter Sweep Engine

**⏱️ Duration**: 45-60 minutes  
**📋 Prerequisites**: Basic familiarity with Portable Alpha concepts  
**🛠️ Tools**: Parameter sweep engine (all 4 modes), Excel output analysis

### Setup

Install the package along with Streamlit and Kaleido so the dashboard and static
exports work:

```bash
pip install -r requirements.txt
pip install streamlit kaleido
```

> **PNG/PDF/PPTX exports require a local Chrome or Chromium installation**.
> On Debian/Ubuntu run `sudo apt-get install -y chromium-browser`.

---

## 🚀 **OVERVIEW: From Single to Multi-Scenario Analysis**

The Portable Alpha Extension Model includes a powerful parameter sweep engine that can analyze hundreds of scenarios simultaneously. This tutorial demonstrates how to leverage this capability for comprehensive analysis across four dimensions:

1. **Capital Allocation Optimization** (`--mode capital`)
2. **Alpha Capture Efficiency** (`--mode alpha_shares`)
3. **Volatility Stress Testing** (`--mode vol_mult`)
4. **Return Sensitivity Analysis** (`--mode returns`)

> **Important**: Every parameter file includes a mandatory
> `analysis_mode` field that tells the CLI which sweep logic to
> apply. The starter templates under `config/` already set this value:
> `params_template.yml` defaults to `returns` for a single scenario,
> while each CSV template specifies the matching sweep mode. The CLI
> will exit with a validation error if `analysis_mode` is missing.

> **Mandatory Metric**: `risk_metrics` must include `ShortfallProb`.
> The CLI aborts if this metric is missing. Legacy files are still
> supported because the Excel exporter and dashboard insert a
> `ShortfallProb` column with `0.0` when absent.

**Professional Impact**: Move from single-point estimates to robust scenario analysis that provides confidence intervals, stress test results, and optimization insights.

---

## 📚 **PART 1: Foundation - Single Scenario Analysis**

### **Objective**: Establish baseline understanding with simple single-scenario run

```bash
# Basic single scenario - foundation for everything that follows
python -m pa_core --params config/params_template.yml --output tutorial_1_baseline.xlsx
```
> **Optional**: Add `--pivot` to include an `AllReturns` sheet for path-based charts. Convert it to Parquet for later tutorials:
> ```bash
> python -m pa_core --params config/params_template.yml --output tutorial_1_baseline.xlsx --pivot
> python - <<PY
> import pandas as pd
> pd.read_excel("tutorial_1_baseline.xlsx", sheet_name="AllReturns").to_parquet("tutorial_1_baseline.parquet")
> PY
> ```

**📊 Key Outputs to Examine**:
- `Output` sheet: Final portfolio metrics 
- `Agent Details` sheet: Individual agent performance
- `Risk Metrics` sheet: Tracking error, Sharpe ratios
- `Capital Allocation` sheet: Funding distribution

**💡 Analysis Questions**:
1. What is the total tracking error for your baseline scenario?
2. Which agent contributes most to portfolio alpha?
3. How much capital is allocated to external alpha vs internal beta?

**⏱️ Expected Time**: 10 minutes

> **Tip**: Add `--dashboard` to automatically open the Streamlit dashboard
> after the run. Include `--png`, `--pdf`, `--pptx`, `--html` or `--gif`
> to export charts during the simulation. Combine these with
> `--alt-text "Description"` for accessible captions.

---

## 📚 **PART 2: Capital Allocation Optimization**

### **Objective**: Find optimal capital allocation across different funding levels

**🎯 Business Question**: "How does portfolio performance change as we scale total capital from $1B to $10B?"

```bash
# Capital mode sweep - 38KB output with optimal allocation analysis
python -m pa_core --mode capital --config my_capital_sweep.yml --output tutorial_1_capital_sweep.xlsx
```

**📈 Analysis Workflow**:

1. **Open Excel Output**: Navigate to `Scenario Analysis` sheet
2. **Capital Range**: Examine scenarios from $1B to $10B total capital
3. **Performance Scaling**: Look for non-linear effects in:
   - Tracking Error vs Capital Size
   - Alpha Generation vs Funding Level  
   - Sharpe Ratio optimization points

**🔍 Key Insights to Find**:
- **Sweet Spot**: What capital level provides optimal risk-adjusted returns?
- **Scaling Limits**: At what point does additional capital hurt performance?
- **Capacity Constraints**: Where do alpha sources become saturated?

**📊 Excel Analysis Tips**:
```excel
# Create pivot charts for:
- Total Capital (X) vs Portfolio Sharpe Ratio (Y)  
- Capital Level (X) vs Tracking Error (Y)
- Funding Size (X) vs Alpha per Dollar (Y)
```

**⏱️ Expected Time**: 15 minutes

---

## 📚 **PART 3: Alpha Capture Efficiency**

### **Objective**: Optimize alpha share allocation across multiple external managers

**🎯 Business Question**: "How should we allocate alpha capture across different external sources to maximize efficiency?"

```bash
# Alpha shares mode - 183KB output, 187 scenario combinations
python -m pa_core --mode alpha_shares --config my_alpha_sweep.yml --output tutorial_1_alpha_sweep.xlsx
```

**📈 Analysis Workflow**:

1. **Scenario Volume**: 187 combinations testing different alpha allocation strategies
2. **Diversification Impact**: Compare concentrated vs diversified alpha strategies
3. **Manager Selection**: Identify optimal combinations of external alpha sources

**🔍 Advanced Analysis**:
- **Alpha Efficiency Frontier**: Plot total alpha captured vs tracking error
- **Concentration Risk**: Find the point where diversification improves risk-adjusted returns
- **Manager Correlation**: Identify which alpha source combinations work best together

**📊 Excel Power Analysis**:
```excel
# Advanced pivot tables:
1. Alpha Source 1 % vs Alpha Source 2 % vs Sharpe Ratio (3D surface)
2. Total Alpha Capture vs Tracking Error (efficiency frontier)  
3. Number of Alpha Sources vs Portfolio Volatility (diversification benefit)
```

**💡 Professional Insights**:
- Optimal alpha share allocation is rarely equal-weight
- Correlation between alpha sources matters more than individual source quality
- There's usually a "sweet spot" for alpha diversification (not too few, not too many)

**⏱️ Expected Time**: 15 minutes

---

## 📚 **PART 4: Volatility Stress Testing**

### **Objective**: Test portfolio resilience across different market volatility environments

**🎯 Business Question**: "How does our portfolio perform during high volatility periods (market stress) vs low volatility environments?"

```bash
# Volatility multiplier mode - 13KB output with stress test scenarios
python -m pa_core --mode vol_mult --config my_vol_sweep.yml --output tutorial_1_vol_sweep.xlsx
```

**📈 Stress Testing Workflow**:

1. **Volatility Range**: Examine scenarios from 0.5x to 3.0x normal market volatility
2. **Risk Management**: Identify when tracking error constraints are breached
3. **Performance Persistence**: Test if alpha generation holds during stress

**🔍 Critical Risk Metrics**:
- **Tracking Error Scaling**: Does TE scale linearly with market volatility?
- **Alpha Erosion**: At what volatility level does alpha generation break down?
- **Sharpe Ratio Resilience**: Which portfolio configurations are most robust?

**⚠️ Risk Management Applications**:
```excel
# Risk dashboards to create:
1. Volatility Multiplier vs Tracking Error (stress test chart)
2. Market Vol Environment vs Portfolio Sharpe (performance persistence)
3. Stress Level vs Alpha Capture Rate (alpha resilience)
```

**💡 Risk Management Insights**:
- Most portfolios show non-linear deterioration above 2x normal volatility
- Alpha sources that work in normal markets may fail during stress
- Optimal position sizing often needs to decrease significantly during high volatility

**⏱️ Expected Time**: 10 minutes

---

## 📚 **PART 5: Return Sensitivity Analysis**

### **Objective**: Understand how portfolio performance varies with different return assumptions

**🎯 Business Question**: "How sensitive are our results to assumptions about internal beta returns and external alpha returns?"

```bash
# Returns mode - comprehensive return sensitivity analysis
python -m pa_core --mode returns --config my_returns_sweep.yml --output tutorial_1_returns_sweep.xlsx
```

> **Note**: A CLI bug in early versions caused this mode to fail. Make
> sure you are running the latest release so the returns sweep
> completes correctly.

**📈 Sensitivity Analysis Workflow**:

1. **Return Assumptions**: Test different combinations of:
   - Internal house returns (beta component)
   - External alpha returns (alpha component)
   - Return volatility assumptions

2. **Assumption Risk**: Identify which return assumptions drive results most

3. **Robustness Testing**: Find scenarios where portfolio strategy breaks down

**🔍 Key Sensitivity Metrics**:
- **Alpha Sensitivity**: How much do results change with alpha return assumptions?
- **Beta Sensitivity**: How important are internal return assumptions?
- **Cross-Sensitivity**: Which combinations of assumptions are most critical?

**📊 Advanced Analytics**:
```excel
# Sensitivity analysis charts:
1. Internal Return (X) vs External Return (Y) vs Sharpe Ratio (Z) - 3D surface
2. Return Assumption Error vs Portfolio Performance Error (robustness)
3. Base Case vs Stress Case Return Assumptions (scenario comparison)
```

**💡 Portfolio Management Insights**:
- Results are usually more sensitive to alpha assumptions than beta assumptions
- Small changes in return assumptions can lead to large changes in optimal allocation
- Robust portfolios perform reasonably well across a range of return scenarios

**⏱️ Expected Time**: 15 minutes

---

## 🎯 **SYNTHESIS: Multi-Scenario Decision Making**

### **Professional Portfolio Management Workflow**

After completing all 5 parts, you now have:

1. **📊 Capital Scaling Analysis**: Optimal size for your strategy
2. **🎯 Alpha Allocation Optimization**: Best manager combination  
3. **⚠️ Stress Test Results**: Risk limits and resilience testing
4. **🔬 Sensitivity Analysis**: Robustness to key assumptions

### **Investment Committee Presentation**

**Executive Summary Template**:
```
Portfolio Strategy Recommendation: [Your recommendation]

Capital Allocation: $X billion optimal size (from Part 2)
Alpha Sources: X% Source A, Y% Source B, Z% Source C (from Part 3)  
Risk Limits: Tracking error <X% under stress (from Part 4)
Key Assumptions: Most sensitive to [assumption] (from Part 5)

Confidence Level: [High/Medium/Low] based on scenario robustness
```

### **Next Steps for Implementation**

1. **Risk Management**: Set position limits based on stress test results
2. **Manager Selection**: Use alpha efficiency results for manager negotiations
3. **Capital Planning**: Plan fundraising based on optimal capacity analysis
4. **Monitoring**: Track key assumptions identified in sensitivity analysis

---

## 🚀 **ADVANCED APPLICATIONS**

### **For Experienced Users**

1. **Custom Parameter Files**: Create your own parameter sweep templates
2. **Excel Automation**: Build automated dashboards from sweep outputs  
3. **Risk Integration**: Combine multiple sweep results for comprehensive risk analysis
4. **Performance Attribution**: Use sweep results for manager performance evaluation

### **Integration with Other Tools**

- **Risk Systems**: Export results to risk management platforms
- **Portfolio Optimization**: Use results as inputs for mean-variance optimization
- **Client Reporting**: Generate client-ready scenario analysis reports

---

## 📋 **SUMMARY: Professional Multi-Scenario Analysis**

**What You Learned**:
✅ How to use all 4 parameter sweep modes  
✅ Professional scenario analysis workflows  
✅ Excel-based results analysis techniques  
✅ Risk management applications  
✅ Investment committee presentation skills  

**Business Impact**:
- Move from single-point estimates to robust scenario analysis
- Provide confidence intervals for all portfolio decisions
- Demonstrate scenario robustness to investment committees  
- Apply professional stress testing methodologies

**Next Tutorial**: Advanced Threshold Analysis - learn to interpret when scenario results indicate problems vs normal variation.

---

*Tutorial 1 Enhanced: Multi-Mode Parameter Sweep Demonstration*  
*Portable-Alpha Extension Model - Professional Edition*
