# Tutorial 2: Advanced Threshold Analysis

**🎯 Goal**: Learn to interpret multi-scenario results and identify parameter combinations that satisfy risk limits.

**⏱️ Duration**: 30-45 minutes
**📋 Prerequisites**: Completion of Tutorial 1
**🛠️ Tools**: Parameter sweep engine, `config_thresholds.yaml`, Excel filters

Whether you worked through just Part&nbsp;1 of Tutorial&nbsp;1 or
ran the full five‑part parameter sweep series, the
metrics discussed here apply to all those results.  The
threshold concepts remain the same regardless of which
analysis mode produced your Excel file.

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

### File selection matrix

Use the correct template and CLI flags for each scenario type:

| Purpose                    | File type | CLI flags                      | Example file                     |
|----------------------------|-----------|--------------------------------|---------------------------------|
| Single scenario (YAML)     | `.yml`    | `--config`                     | `params_template.yml`           |
| Single scenario            | `.yml`    | `--config`                     | `params_template.yml`           |
| Capital allocation sweep   | `.yml`    | `--config` + `--mode capital`   | Copy `params_template.yml`, set `analysis_mode: capital` |
| Alpha shares optimization  | `.yml`    | `--config` + `--mode alpha_shares` | Copy `params_template.yml`, set `analysis_mode: alpha_shares` |
| Returns sensitivity        | `.yml`    | `--config` + `--mode returns`   | Copy `params_template.yml`, set `analysis_mode: returns` |
| Volatility stress test     | `.yml`    | `--config` + `--mode vol_mult`  | Copy `params_template.yml`, set `analysis_mode: vol_mult` |

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
   Shortfall probability below **5%** is typically considered healthy, so use
   the baseline results to practise bringing both metrics back within these
   limits.

## 📚 **PART B – Capital Allocation Sweep**

Use the sweep engine to test funding levels automatically:

```bash
python -m pa_core.cli \
  --config my_capital_sweep.yml \
  --mode capital \
  --index sp500tr_fred_divyield.csv \
  --output Tutorial2_CapitalSweep.xlsx \
  --dashboard
```

The `--dashboard` flag launches the Streamlit interface automatically so you can
review scenarios side by side without running a separate command.

Sort the **Summary** sheet by `TE` or `ShortfallProb` and filter for
`TE < 0.03` and `ShortfallProb < 0.05` to find healthy scenarios.

## 📚 **PART C – Advanced Sweeps**

Run additional sweeps and keep the results separate:

```bash
python -m pa_core.cli --config my_alpha_sweep.yml --mode alpha_shares --index sp500tr_fred_divyield.csv --output Tutorial2_AlphaSweep.xlsx
python -m pa_core.cli --config my_vol_sweep.yml --mode vol_mult --index sp500tr_fred_divyield.csv --output Tutorial2_VolSweep.xlsx \
  --dashboard
```

Launching the dashboard after each run helps you compare sweeps interactively without reopening the files manually.

Each file may contain dozens or even hundreds of scenarios (50–200 depending on
the template size). Open the **Summary** sheet or load the workbook in the
dashboard to browse them via the **Scenario** dropdown. Apply the same
threshold checks as in Part B.

## 📚 **PART D – Multi‑Scenario Interpretation**

1. **Compare to thresholds** – use `config_thresholds.yaml` to highlight breaches. Values
   below **3% TE** and **5% ShortfallProb** are generally considered healthy.
2. **Filter combinations** – in the **Summary** sheet, filter for `TE < 0.03` and `ShortfallProb < 0.05`.
3. **Pivot analysis** – create a pivot table of `AnnReturn`, `TE` and `ShortfallProb` by scenario name.
4. **Iterate quickly** – adjust parameters and rerun sweeps until most scenarios fall in the green zone.
5. **Document findings** – record which parameter sets meet all limits and note any persistent issues.

## 📈 **Interpreting Breaches and Next Actions**

When `TE` or `ShortfallProb` exceeds the limits defined in `config_thresholds.yaml` treat the result as a warning sign rather than a hard failure. Use the sweep outputs to diagnose which parameters drive the breach.

**Common adjustments**:
* Lower the `active_share` or external capital to reduce tracking error.
* Increase diversification across agents to lower shortfall risk.
* Re‑run the relevant sweep with narrower parameter ranges to zero in on compliant combinations.

Document the scenario labels that remain in violation so you can revisit them after adjusting the assumptions or thresholds.

---

**Next Tutorial**: Learn to visualise these results interactively in Tutorial 3.

*Tutorial 2 Enhanced: Advanced Threshold Analysis*
