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
  --png --gif
```

The command produces `plots/fan.png` and `plots/fan.gif`. GIF creation logs a warning if Chrome or Kaleido is missing.

### 3. Recommended Workflow

1. Run a sweep with `--output` so each scenario is saved.
2. Convert the `AllReturns` sheet to Parquet for path‑based charts.
3. Iterate over the summary table or call `scripts/visualise.py` repeatedly to build a gallery.
4. Review the images in `plots/` or assemble them into a presentation deck.

---

*Tutorial 5 Enhanced: Automated Bulk Visualization Workflows*
