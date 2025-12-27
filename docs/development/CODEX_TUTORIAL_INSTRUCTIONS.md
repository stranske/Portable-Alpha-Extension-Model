# CODEX IMPLEMENTATION INSTRUCTIONS: Tutorial 1 Restructure

## 🎯 **TASK OVERVIEW**
**DEPENDENCY NOTICE**: This tutorial restructuring task is **dependent on parameter sweep implementation**. The tutorial restructure should be performed AFTER the parameter sweep engine is implemented and working.

**Current Status**: Tutorial 1 restructuring can proceed as documented, but the full 5-part structure assumes working parameter sweep functionality that will be implemented separately.

Restructure Tutorial 1 in `docs/UserGuide.md` from a single-scenario tutorial to a comprehensive 5-part introduction that covers basic operation plus all four parameter sweep modes.

## 📁 **FILES TO MODIFY**
- **Primary**: `docs/UserGuide.md` (lines ~91-110, "Introductory Tutorial 1" section)
- **Reference**: Use existing CSV templates in `config/` folder
- **Testing**: Verify commands work with existing setup

## 📋 **DETAILED IMPLEMENTATION STEPS**

### **Step 1: Update Tutorial Roadmap** 
**Location**: `docs/UserGuide.md` around line 66
**Action**: Update the tutorial roadmap to reflect new 5-part Tutorial 1

**Current**:
```markdown
1. **Introductory Tutorial 1 – Implement a Scenario** – run the simulation from a parameter file and produce `Outputs.xlsx`.
```

**Replace With**:
```markdown
1. **Introductory Tutorial 1 – Master the Program (5 Parts)**
   - **Part 1**: Basic Program Operation - single scenario fundamentals
   - **Part 2**: Capital Mode - allocation percentage sweeps
   - **Part 3**: Returns Mode - return/volatility sensitivity analysis  
   - **Part 4**: Alpha Shares Mode - alpha/beta split optimization
   - **Part 5**: Vol Mult Mode - volatility stress testing
```

### **Step 2: Replace Tutorial 1 Section**
**Location**: `docs/UserGuide.md` starting around line 91
**Action**: Replace the existing "Introductory Tutorial 1" section with the new 5-part structure

**New Content Structure**:

```markdown
### Introductory Tutorial 1 – Master the Program (5 Parts)

This comprehensive tutorial introduces you to both basic operation and the powerful parameter sweep capabilities. Work through all five parts in order to build complete understanding.

#### Part 1: Basic Program Operation

**Objective**: Run your first simulation and understand the fundamental output structure.

As a new user, start with the simplest possible command to establish baseline understanding:

1. **Copy the basic template**:
   ```bash
   cp config/params_template.yml my_first_scenario.yml
   ```

2. **Run your first simulation** (single scenario, no sweep):
   ```bash
   python -m pa_core.cli \
     --config my_first_scenario.yml \
     --index sp500tr_fred_divyield.csv \
     --output MyFirstResults.xlsx
   ```

3. **Understand the console output**: You'll see a Rich table showing:
   - `AnnReturn`: Annualized return percentage for each sleeve
   - `AnnVol`: Annualized volatility (risk measure)
   - `VaR`: Value at Risk at 5% level
   - `BreachProb`: Probability of funding shortfall
   - `TE`: Tracking Error relative to benchmark

4. **Examine the Excel file**: Open `MyFirstResults.xlsx` to see:
   - **Summary Sheet**: Key metrics for all sleeves
   - **Inputs Sheet**: Confirms your configuration parameters
   - **Risk-Return Chart**: Visual representation embedded in Excel

**Success Check**: You should see results for 3-4 sleeves (Internal PA, External PA, Active Extension, etc.) with realistic financial metrics.

#### Part 2: Capital Mode - Allocation Sweeps

**Objective**: Understand how parameter sweeps work by varying capital allocations.

The `--mode=capital` parameter runs multiple scenarios automatically, varying external and active extension capital allocations:

1. **Create a capital sweep config** by copying `params_template.yml`:
   ```bash
   cp config/params_template.yml my_capital_sweep.yml
   # Edit to set analysis_mode: capital and add capital ranges
   ```

2. **Run a capital allocation sweep**:
   ```bash
   python -m pa_core.cli \
     --config my_capital_sweep.yml \
     --index sp500tr_fred_divyield.csv \
     --mode capital \
     --output CapitalSweep.xlsx
   ```

3. **Compare the results**: Notice that `CapitalSweep.xlsx` now contains:
   - Multiple sheets (one per allocation scenario)
   - Summary sheet with all combinations
   - Risk-return chart showing the efficient frontier

**Key Insight**: Capital mode helps you find optimal allocation percentages by testing multiple combinations automatically.

#### Part 3: Returns Mode - Sensitivity Analysis

**Objective**: Explore how different return and volatility assumptions affect outcomes.

Use `--mode=returns` to test various return/volatility scenarios:

1. **Create a returns sweep config** by copying `params_template.yml`:
   ```bash
   cp config/params_template.yml my_returns_sweep.yml
   # Edit to set analysis_mode: returns and add return/vol ranges
   ```

2. **Run returns sensitivity analysis**:
   ```bash
   python -m pa_core.cli \
     --config my_returns_sweep.yml \
     --index sp500tr_fred_divyield.csv \
     --mode returns \
     --output ReturnsSweep.xlsx
   ```

3. **Interpret the results**: The sweep shows how sensitive your strategy is to return assumptions. Higher expected returns generally increase both returns and risks.

**Key Insight**: Returns mode helps stress-test your assumptions about future market performance.

#### Part 4: Alpha Shares Mode - Optimization

**Objective**: Understand alpha vs. beta share allocation and its impact on tracking error.

Use `--mode=alpha_shares` to optimize the split between alpha-generating and beta-matching components:

1. **Create an alpha shares sweep config** by copying `params_template.yml`:
   ```bash
   cp config/params_template.yml my_alpha_sweep.yml
   # Edit to set analysis_mode: alpha_shares and add alpha/beta ranges
   ```

2. **Run alpha/beta optimization**:
   ```bash
   python -m pa_core.cli \
     --config my_alpha_sweep.yml \
     --index sp500tr_fred_divyield.csv \
     --mode alpha_shares \
     --output AlphaSweep.xlsx
   ```

3. **Analyze tracking error trade-offs**: Higher alpha allocation may increase returns but also tracking error.

**Key Insight**: Alpha shares mode helps balance return enhancement with tracking error constraints.

#### Part 5: Vol Mult Mode - Stress Testing

**Objective**: Perform comprehensive stress testing by scaling volatilities.

Use `--mode=vol_mult` to test how your strategy performs under different volatility regimes:

1. **Create a vol mult sweep config** by copying `params_template.yml`:
   ```bash
   cp config/params_template.yml my_vol_sweep.yml
   # Edit to set analysis_mode: vol_mult and add multiplier ranges
   ```

2. **Run volatility stress test**:
   ```bash
   python -m pa_core.cli \
     --config my_vol_sweep.yml \
     --index sp500tr_fred_divyield.csv \
     --mode vol_mult \
     --output VolStressTest.xlsx
   ```

3. **Evaluate resilience**: See how your strategy performs in low, normal, and high volatility environments.

**Key Insight**: Vol mult mode reveals how robust your strategy is to changing market volatility.

#### Tutorial 1 Summary

You've now mastered:
- ✅ Basic single-scenario operation (Part 1)
- ✅ Capital allocation optimization (Part 2)  
- ✅ Return assumption sensitivity (Part 3)
- ✅ Alpha/beta split optimization (Part 4)
- ✅ Volatility stress testing (Part 5)

**Next Steps**: Proceed to Tutorial 2 to learn detailed metric interpretation, or Tutorial 3 to explore the interactive dashboard. All visualization features work with results from any of these five approaches.

**Troubleshooting**:
- If commands fail, ensure you're in the correct directory and virtual environment is activated
- Check that CSV templates exist in the `config/` folder
- Verify `sp500tr_fred_divyield.csv` is present in the root directory
- Use `python -m pa_core.cli --help` to see all available options
```

### **Step 3: Update Cross-References**
**Action**: Update Tutorial 2 and 3 introductions to reference the new 5-part Tutorial 1

**Tutorial 2 Update** (around line 111):
```markdown
### Introductory Tutorial 2 – Interpret the Metrics (Risk/Return, Shortfall and Tracking Error)

This tutorial explains how to read the results produced in Tutorial 1 (any of the 5 parts). Whether you ran a single scenario (Part 1) or parameter sweeps (Parts 2-5), the core metrics remain the same...
```

**Tutorial 3 Update** (around line 127):
```markdown
### Introductory Tutorial 3 – Visualise the Results (Dashboard and Scripts)

This tutorial shows how to visualise the metrics produced in Tutorial 1 (all 5 parts) and Tutorial 2. The dashboard works with results from any mode - single scenarios, capital sweeps, returns analysis, alpha optimization, or volatility stress tests...
```

## ✅ **QUALITY CHECKLIST**
- [ ] All 5 commands are copy-pasteable and work correctly
- [ ] Each part builds logically on the previous part
- [ ] Template file references are accurate (`config/*.csv`)
- [ ] Console output expectations are set appropriately  
- [ ] Excel file structure is explained for each mode
- [ ] Cross-references to Tutorials 2-3 are updated
- [ ] Troubleshooting section covers common issues
- [ ] Markdown formatting is consistent with existing style

## 🎯 **SUCCESS CRITERIA**
1. **Beginner-Friendly**: A complete newcomer can follow Part 1 and succeed
2. **Progressive Learning**: Each part introduces one new concept
3. **Mode Mastery**: User understands when and why to use each mode
4. **Self-Sufficient**: User can troubleshoot and adapt commands
5. **Integration Ready**: Prepares user for advanced tutorials

## 📝 **TESTING REQUIREMENTS**
After implementation, verify:
- All commands execute without errors
- Referenced files exist (`config/*.csv`, `sp500tr_fred_divyield.csv`)  
- Excel outputs match descriptions
- Tutorials 2-3 flow naturally from new Tutorial 1 structure
