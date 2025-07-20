# Tutorial 2: Advanced Threshold Analysis

**🎯 Goal**: Learn to interpret multi-scenario results and identify parameter combinations that satisfy risk limits.

**⏱️ Duration**: 30-45 minutes
**📋 Prerequisites**: Completion of Tutorial 1
**🛠️ Tools**: Parameter sweep engine, `config_thresholds.yaml`, Excel filters

### Setup

Ensure all dependencies are installed so the CLI and dashboard work:

```bash
pip install -r requirements.txt
pip install streamlit kaleido
```

> **Tip**: Use a local Chrome/Chromium browser for PNG/PDF exports.

> **Mandatory Metric**: Ensure your parameter file lists `ShortfallProb`
> under `risk_metrics`. The CLI stops with an error otherwise, but the
> exporter inserts a default `0.0` column so older files still work.

---

## 📚 **PART A – Single Scenario Review**

1. Copy the baseline template:
   ```bash
   cp config/params_template.yml my_threshold_test.yml
   ```
2. Run a single scenario and save the results:
   ```bash
   python -m pa_core.cli \
     --config my_threshold_test.yml \
     --index sp500tr_fred_divyield.csv \
     --output Tutorial2_Baseline.xlsx
   ```
3. Open `Tutorial2_Baseline.xlsx` and inspect `ShortfallProb` and `TE`.
   The sample data intentionally breaches the **3%** tracking-error budget.

## 📚 **PART B – Capital Allocation Sweep**

Use the sweep engine to test funding levels automatically:

```bash
python -m pa_core.cli \
  --params config/capital_mode_template.csv \
  --mode capital \
  --index sp500tr_fred_divyield.csv \
  --output Tutorial2_CapitalSweep.xlsx
```

Sort the **Summary** sheet by `TE` or `ShortfallProb` to find compliant scenarios.

## 📚 **PART C – Advanced Sweeps**

Run additional sweeps and keep the results separate:

```bash
python -m pa_core.cli --params config/alpha_shares_mode_template.csv --mode alpha_shares --index sp500tr_fred_divyield.csv --output Tutorial2_AlphaSweep.xlsx
python -m pa_core.cli --params config/vol_mult_mode_template.csv --mode vol_mult --index sp500tr_fred_divyield.csv --output Tutorial2_VolSweep.xlsx
```

Each file may contain dozens of scenarios. Apply the same threshold checks as in Part B.

## 📚 **PART D – Multi‑Scenario Interpretation**

1. **Compare to thresholds** – use `config_thresholds.yaml` to highlight breaches.
2. **Filter combinations** – in the **Summary** sheet, filter for `TE < 0.03` and low `ShortfallProb`.
3. **Pivot analysis** – create a pivot table of `AnnReturn`, `TE` and `ShortfallProb` by scenario name.
4. **Iterate quickly** – adjust parameters and rerun sweeps until most scenarios fall in the green zone.
5. **Document findings** – record which parameter sets meet all limits and note any persistent issues.

---

**Next Tutorial**: Learn to visualise these results interactively in Tutorial 3.

*Tutorial 2 Enhanced: Advanced Threshold Analysis*
