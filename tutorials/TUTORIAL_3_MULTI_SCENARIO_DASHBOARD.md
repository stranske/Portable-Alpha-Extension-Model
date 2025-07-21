# Tutorial 3: Multi-Scenario Dashboard Workflows

**🎯 Goal**: Visualise parameter sweep results and compare up to 200 scenarios in the Streamlit dashboard.

**⏱️ Duration**: 30-45 minutes
**📋 Prerequisites**: Completion of Tutorial 2, working parameter sweep outputs
**🛠️ Tools**: Streamlit dashboard, parameter sweep engine, Chrome/Kaleido for exports

You can load results from any part of Tutorial&nbsp;1.
Single scenarios and all sweep modes produce workbooks that
the dashboard reads in the same way.

### Setup

Install the extra packages if they are not already available:

```bash
pip install streamlit kaleido
sudo apt-get install -y chromium-browser  # required for PNG/PDF downloads
```

### 1. Generate a Parameter Sweep

Run a sweep so you have a multi-scenario Excel file to explore. Any mode works.
Include the `--pivot` flag so an `AllReturns` sheet is written for advanced charts.
Example using capital mode:

```bash
python -m pa_core.cli \
  --params config/capital_mode_template.csv \
  --mode capital \
  --output DashboardSweep.xlsx \
  --pivot \
  --dashboard
```

The `--dashboard` flag launches the Streamlit app automatically after the run completes.

### 2. Load Results

1. In the sidebar, enter the path to `DashboardSweep.xlsx`.
2. If you saved an `AllReturns` sheet during the run, convert it to Parquet and place `DashboardSweep.parquet` in the same folder. This file unlocks the **Funding fan** and **Path dist** tabs, which rely on the full return paths.
3. Use the **Scenario** dropdown to browse each parameter combination. Files in the 38‑183 KB range load quickly and may contain up to 200 scenarios.

### 3. Navigate the Dashboard

- **Risk-return** – scatter plot with tracking‑error cap and excess‑return lines from `config_thresholds.yaml`.
- **Funding fan** – median path with confidence bands (requires Parquet data).
- **Path dist** – histogram of final returns with optional CDF view.
- **Diagnostics** – table of raw metrics for the selected scenario.

Start on **Risk-return** to see the full scenario landscape.  Use the **Scenario** dropdown to focus on individual combinations and then flip to **Funding fan** or **Path dist** to inspect the return paths.  The **Diagnostics** tab lists every metric so you can confirm threshold compliance.

Tick **Auto-refresh** to reload the results periodically while a long sweep runs. Two download buttons save the headline PNG chart and the Excel file. PNG and PDF exports require Chrome/Chromium plus Kaleido; without them the buttons fail silently.

### 4. Analyse Multiple Scenarios

Compare scenarios by switching the dropdown. Look for clusters that meet your tracking‑error budget and have low `ShortfallProb`. The colour‑blind‑safe palette highlights compliant cases automatically. Use the dashboard together with the summary table in Excel to identify the most promising parameter sets.

### 5. When to Use Scripts Instead

For bulk chart generation outside the dashboard, call `scripts/visualise.py` on the same output file:

```bash
python scripts/visualise.py \
  --plot risk_return \
  --xlsx DashboardSweep.xlsx \
  --png --alt-text "Risk-return" 
```

This produces static images under `plots/`. Use the dashboard for interactive exploration and the script for automated exports.

**Next Tutorial**: Professional Bulk Export Workflows – learn to generate presentation-ready charts from parameter sweeps.

---

*Tutorial 3 Enhanced: Multi-Scenario Dashboard Workflows*
