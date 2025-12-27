# Tutorial 8: Automated Stress Testing

**🎯 Goal**: Replace manual scenario checks with systematic parameter sweeps.

**⏱️ Duration**: 20 minutes
**📋 Prerequisites**: Completion of Tutorial 7
**🛠️ Tools**: Parameter sweep engine, dashboard, `viz.delta_heatmap`

### Setup

Install Streamlit and Kaleido plus a local Chrome or Chromium browser so the dashboard and static exports work:

```bash
pip install streamlit kaleido
sudo apt-get install -y chromium-browser
```

### Step 1 – Run a stress‑test sweep

Create a sweep config from `params_template.yml` and always pass `--output` so previous results are preserved:

```bash
python -m pa_core.cli \
  --mode returns \
  --config my_returns_sweep.yml \
  --output StressTest.xlsx
```

### Step 2 – Review the results

Open the `Summary` sheet or load the file in the dashboard. The **Scenario** dropdown lets you compare up to 200 combinations. Filter by `TE` and `ShortfallProb` to find compliant cases.

### Step 3 – Compare sweeps directly

Generate a diverging‑colour grid to see how one run differs from another:

```python
import pandas as pd
from pa_core.viz import delta_heatmap

base = pd.read_excel("StressTest_base.xlsx", sheet_name="Summary")
alt = pd.read_excel("StressTest_alt.xlsx", sheet_name="Summary")
fig = delta_heatmap.make(base, alt, value="Sharpe")
fig.write_image("plots/delta_heatmap.png")
```

---

**Next Tutorial**: Enhanced Export Bundle Integration – archive full sets of figures with one call.

*Tutorial 8 Enhanced: Automated Stress Testing*
