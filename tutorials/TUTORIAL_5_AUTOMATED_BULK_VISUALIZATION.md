# Tutorial 5: Automated Bulk Visualization Workflows

**🎯 Goal**: Generate galleries of charts from parameter sweep outputs using `scripts/visualise.py`.

**⏱️ Duration**: 20-30 minutes
**📋 Prerequisites**: Completion of Tutorial 4 and parameter sweep results with `--pivot` data
**🛠️ Tools**: `scripts/visualise.py`, Parquet path data, Chrome/Chromium + Kaleido

### Setup

Ensure `Outputs.xlsx` and the matching `Outputs.parquet` are in the same folder. Convert the `AllReturns` sheet after running the CLI if needed:

```python
import pandas as pd
df = pd.read_excel("Outputs.xlsx", sheet_name="AllReturns")
df.to_parquet("Outputs.parquet")
```
This Parquet file unlocks path-based plots such as `fan` and `path_dist`.

Sweeps typically contain **50–200 scenarios**, so these scripts are designed to
loop over large summary tables automatically. Each scenario can generate one or
more figures without manual intervention.

Install **Kaleido** and a local Chrome or Chromium browser so PNG, PDF and PPTX exports work:

```bash
pip install kaleido
sudo apt-get install -y chromium-browser
```
### Available Plot Types
| Plot Name | When to Use It |
|-----------|----------------|
| `risk_return` | Compare annualised return vs volatility across scenarios. |
| `fan` | Funding fan showing median path and confidence ribbon. Requires `Outputs.parquet`. |
| `path_dist` | Histogram of monthly returns sourced from `Outputs.parquet`. |
| `corr_heatmap` | Highlight monthly return correlations between agents. |
| `sharpe_ladder` | Rank scenarios by Sharpe ratio. |
| `rolling_panel` | Rolling drawdown, tracking error and Sharpe metrics. |
| `surface` | 3‑D risk/return surface from parameter sweeps. |
| `category_pie` | Donut summarising capital allocation by agent. |
| `overlay` | Line chart comparing cumulative paths across agents. |
| `waterfall` | Step-by-step bar chart of capital contributions. |
| `gauge` | Dial showing risk or return relative to a target. |
| `moments_panel` | Rolling mean, volatility and skew dashboard. |
| `rolling_corr_heatmap` | Heatmap revealing correlation drift over time. |
| `weighted_stack` | Stacked bars scaled by horizon length. |


### 1. Create a Gallery of Charts

Loop over the summary table and save one figure per scenario:

```python
import pandas as pd
from pa_core.viz import risk_return, export_bundle

df = pd.read_excel("Sweep.xlsx", sheet_name="Summary")
for label, df_scenario in df.groupby("Scenario"):
    fig = risk_return.make(df_scenario)
    export_bundle.save([fig], f"plots/{label}")
```

### 2. Batch Export with scripts/visualise.py

Run the helper script for each plot type you need:

```bash
python scripts/visualise.py \
  --plot fan \
  --xlsx Sweep.xlsx \
  --parquet Sweep.parquet \
  --png --gif --alt-text "Funding fan"
```
The command produces `plots/fan.png` and `plots/fan.gif`. GIF creation logs a warning if Chrome or Kaleido is missing. PNG/PDF/PPTX exports behave the same way -- check the console logs if no file appears. Use `--alt-text` to provide an accessible description for HTML or PPTX files.

### 3. Recommended Workflow

1. Run a sweep with `--output` so each scenario is saved.
2. Convert the `AllReturns` sheet to Parquet for path‑based charts.
3. Iterate over the summary table or call `scripts/visualise.py` repeatedly to build a gallery.
4. Review the images in `plots/` or assemble them into a presentation deck.

---

**Next Tutorial**: Dynamic Agent Configuration – add and test your own sleeve classes.
*Tutorial 5 Enhanced: Automated Bulk Visualization Workflows*
