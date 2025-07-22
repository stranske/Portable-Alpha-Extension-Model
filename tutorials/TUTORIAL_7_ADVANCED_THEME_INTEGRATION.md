# Tutorial 7: Advanced Theme Integration

**🎯 Goal**: Apply custom colour schemes and threshold settings across parameter sweeps.

**⏱️ Duration**: 20 minutes
**📋 Prerequisites**: Completion of Tutorial 6
**🛠️ Tools**: `config_theme.yaml`, `config_thresholds.yaml`, `pa_core.viz.theme`

### Setup

Install Streamlit and Kaleido plus a local Chrome or Chromium browser so the dashboard and static exports work:

```bash
pip install streamlit kaleido
sudo apt-get install -y chromium-browser
```

### Step 1 – Edit the theme files

Adjust colours and fonts in `config_theme.yaml` and set threshold levels in `config_thresholds.yaml`:

```yaml
# config_thresholds.yaml
shortfall_green: 0.05
shortfall_amber: 0.10
```

Reload these files before generating figures:

```python
from pa_core.viz import theme

theme.load_theme("my_theme.yaml")
```

### Step 2 – Generate styled charts

Run a sweep and apply your theme when creating figures:

```python
import pandas as pd
from pa_core.viz import risk_return

theme.load_theme("my_theme.yaml")
df = pd.read_excel("Sweep.xlsx", sheet_name="Summary")
fig = risk_return.make(df)
fig.write_image("plots/sweep.png")
```

Loop over multiple theme files to style each scenario differently.

---

**Next Tutorial**: Automated Stress Testing – replace manual checks with systematic sweeps.

*Tutorial 7 Enhanced: Advanced Theme Integration*
