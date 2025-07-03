# Agents Architecture Guide  
*Portable‑Alpha + Active‑Extension Model*

> “Make everything as simple as possible, but no simpler.” – A. Einstein  
> (…who probably never had to vectorize Monte‑Carlo code, but would have sympathised.)

---

## 1  Why “Agents”?

The existing notebook (`Portable_Alpha_Vectors.ipynb`) bundles business logic, UI, simulation, and reporting into one monolith.  Splitting each capital sleeve into an **agent**:

* **Encapsulates** its parameters and maths behind a clean interface.  
* **Vectorises** return generation (NumPy ops on `shape = (n_sim, n_months)` arrays).  
* **Parametrises** behaviour via plain‑text config—trivial to grid‑search in CI.  
* **Enables** drop‑in replacement (e.g. a new “Overlay Options” sleeve) without touching the Monte‑Carlo driver.

---

## 2  Proposed Package Layout

pa_core/
│
├─ agents/ # one file per agent class
│ ├─ base.py # In‑house alpha + beta
│ ├─ external_pa.py # External PA sleeve
│ ├─ active_ext.py # 150/50 etc.
│ ├─ internal_beta.py # Margin sleeve (β – f)
│ ├─ internal_pa.py # Pure in‑house α
│ └─ registry.py # Factory to build agents from YAML/CSV rows
│
├─ data/ # CSV, parquet, & download helpers
│
├─ sim/ # vectorised MC engine
│ ├─ covariance.py # Σ builder from vols & ρs
│ ├─ paths.py # batched MVN + financing draws
│ └─ metrics.py # return → TE, VaR, breach, …
│
├─ reporting/
│ ├─ excel.py # Outputs.xlsx writer
│ └─ console.py # CLI pretty‑print
│
├─ cli.py # python -m pa_core --params …
└─ config.py # dataclass wrappers around YAML/CSV
// NEW – visualisation & dashboard layer
├─ viz/                       # Pure‑function charts (Plotly)
│   ├─ __init__.py
│   ├─ theme.py               # colour‑blind‑safe template
│   ├─ risk_return.py         # bubble chart
│   ├─ fan.py                 # 95 % confidence ribbon fan
│   ├─ path_dist.py           # histogram + CDF
│   ├─ corr_heatmap.py        # monthly‑return corr
│   └─ sharpe_ladder.py       # sorted bar of Sharpe
│
├─ dashboard/                 # Streamlit front‑end
│   └─ app.py
│
├─ scripts/
│   └─ visualise.py           # CLI wrapper → PNG/PDF/PPTX

---

## 3  Agent Interface

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray  # shape: (n_sim, n_months)

@dataclass
class AgentParams:
    name: str
    capital_mm: float      # X, Y, Z, or W
    beta_share: float      # portion applied to (r_β – f_t)
    alpha_share: float     # portion applied to stream‑specific α
    extra_args: dict       # e.g. θ_ExtPA, active_share S, …

class Agent:
    """Abstract sleeve.  Child classes implement `monthly_returns`."""
    def __init__(self, p: AgentParams):
        self.p = p

    def monthly_returns(
        self,
        r_beta: Array,
        alpha_stream: Array,
        financing: Array,
    ) -> Array:
        raise NotImplementedError
```

Concrete subclasses override monthly_returns. Each subclass must:

Vectorise: never loop over simulations or months.

Dimension‑check inputs.

Return an (n_sim, n_months) array of raw monthly sleeve returns.

3.1 Concrete Agents
| Class                  | α‑stream    | β treatment    | Key params              |
| ---------------------- | ----------- | -------------- | ----------------------- |
| `BaseAgent`            | In‑house H  | `(wβ)(rβ – f)` | `w_beta_H`, `w_alpha_H` |
| `ExternalPAAgent`      | Manager M   | fraction of X  | `θ_ExtPA`               |
| `ActiveExtensionAgent` | Extension E | fraction of Y  | `active_share` (S)      |
| `InternalBetaAgent`    | –           | **all** β      | Margin sleeve (W)       |
| `InternalPAAgent`      | In‑house H  | 0              | Z                       |

// NEW – Category legend for downstream charts
# Category map (used by viz.theme)
CATEGORY_BY_AGENT = {
    "InternalPAAgent":      "Internal Portable Alpha",
    "InternalBetaAgent":    "Internal Portable Alpha",
    "ExternalPAAgent":      "External Portable Alpha",
    "ActiveExtensionAgent": "External Portable Alpha",
    "BaseAgent":            "Benchmark / Passive",
}

(This covers everything in the README spec. Adding a new agent = subclass + registry entry.)

4  Simulation Engine (sim/)
1. Draw paths
paths.py calls NumPy’s multivariate_normal once per sleeve group—not per agent—to create four rank‑3 tensors: r_beta, r_H, r_E, r_M. Financing draw is a separate tensor with spike logic vectorised.

2. Dispatch
The Monte‑Carlo driver iterates agents, not sims:

for agent in registry.build_all(config):
    sleeve_returns[agent.p.name] = agent.monthly_returns(
        r_beta, stream_map[agent], financing
    )
3. Aggregate
metrics.py handles compounding and analytics in pure NumPy; pandas DataFrames are only used at the very end for pretty output.

5  Parameterisation
Friendly CSV stays supported for the Excel crowd.

Internally converted to a typed ModelConfig dataclass (see config.py).

Alternate YAML file accepted (--params foo.yaml) to remove the Excel‑style header limitations.

Validation via pydantic‑v2 to catch nonsense (e.g. W + Z > total_capital).

// NEW –‑export & dashboard flags
@dataclass
class RunFlags:
    save_xlsx: str | None = "Outputs.xlsx"
    png: bool = False
    pdf: bool = False
    pptx: bool = False
    launch_dashboard: bool = False   # streamlit run after sim



6  Vectorisation Checklist
| Item                  | Status               | To‑do                                                                    |
| --------------------- | -------------------- | ------------------------------------------------------------------------ |
| **Covariance build**  | ✅ existing           | Move to `covariance.py`; ensure symmetric σ clipping                     |
| **MVN sampling**      | ⚠️ loops in notebook | Replace with single `np.random.default_rng().multivariate_normal()` call |
| **Financing spikes**  | ⚠️ loop              | Use `rng.uniform(size) < p` mask                                         |
| **Per‑agent returns** | ⚠️ loop over months  | Compute in closed‑form array ops                                         |
| **Metrics**           | partial              | Standardise in `metrics.py`                                              |
// NEW
| Chart functions vectorised | ✅ viz package | Pure Plotly helpers under `viz/` |


7  Reporting
// NEW

Excel – `reporting/excel.py` now calls `viz.risk_return.make()` and embeds the PNG directly in the ``Summary`` sheet using ``openpyxl``.

Static exports – scripts/visualise.py accepts --png --pdf --pptx; each flag triggers fig.write_image() or python‑pptx helper.

Dashboard – `dashboard/app.py` caches loaded data with ``@st.cache_data`` and offers an optional auto‑refresh checkbox. Launch with ``--launch_dashboard`` or run ``streamlit run dashboard/app.py``.


8  Testing & CI
Unit tests in tests/ (pytest):

covariance symmetry,

agent maths identity vs. hand‑calc small matrix,

VaR/TE edge cases.

Property‑based tests (Hypothesis) on the agent interface: random params → returns matrix shape & finite values.

GitHub Actions: lint (ruff), type‑check (pyright), tests.

9  Performance Targets
| Dimension            | Goal               | Rationale                    |
| -------------------- | ------------------ | ---------------------------- |
| 100 k sims × 12 m    | < 3 s on M2 laptop | Enough for grid search in CI |
| Memory               | < 2 GB             | Fits Codespaces free tier    |
| **Vectorised** loops | 0                  | Loops only at agent registry |

10  Extensibility Playbook
Add a new sleeve

Create agents/new_sleeve.py subclass.

Register in registry.py.

Add CLI flag if params differ.

Swap α‑stream source
Just point the agent at a different slice of the drawn MVN tensor or supply a callback.

Switch to GPU
Replace NumPy import with CuPy behind a --backend flag; interface untouched.

11  Glossary
Sleeve / Agent – A self‑contained capital bucket with its own return equation.

Path – A (n_months,) vector of returns for one simulation.

Tensor – A stacked array of paths; here (n_sim, n_months).

Registry – Factory that turns AgentParams into concrete agent objects.

---

## Next steps & open questions

1. **Parameter defaults** – confirm your latest assumptions (e.g. did rho_E_M drift to 0.05?).  
2. **Financing spikes per sleeve** – current notebook applies identical spike logic across all sleeves; do you want differentiated parameters for Internal vs. External financing?  
3. **Random‑seed strategy** – single global RNG or per‑agent sub‑streams (could aid reproducibility when sleeves are added/removed).  
4. **Outputs.xlsx layout** – retain current sheet order or collapse into one pivot‑table‑ready sheet?

Kick back any tweaks; happy to iterate.

---

*(Caveat: Some internal package names may differ slightly from the current repo tree—rename to taste.  File/line references in the spec come from the public README and notebook as of 29 Jun 2025.)*


# Agents – How to run, tune & read them ⚙️📊

> *“Forecasts may be wrong, but traffic‑lights should never be vague.”*

---

## 1. Quick‑start for PMs & Ops 🏃‍♂️💨

```bash
# Install (one‑time)
pip install -r requirements.txt         # pandas, numpy, plotly, streamlit, xlsxwriter, kaleido …

# Run a 500‑path, 15‑year simulation of all agents and save outputs
python main.py \
  --agent ExternalPA ActiveExt InternalPA InternalBeta Base \
  --n_sims 500 --n_months $((15*12)) \
  --save_xlsx Outputs.xlsx \
  --seed 42


```

```python
# pa_core/agents/my_new_agent.py
from pa_core.agents.base import BaseAgent

class MyNewAgent(BaseAgent):
    def monthly_returns(self, r_beta, alpha_stream, financing):
        # 1 Generate monthly returns
        # 2 Return np.ndarray shape (n_sim, n_months)

```
## 12  Visual‑Analytics Subsystem

**Goal** – Serve PMs & Ops an interactive narrative focussed on funding‑shortfall risk, draw‑down control, and TE trade‑offs.

### 12.1  Core chart contracts (`viz/`)

| Function            | Input (pandas)                                | Output | Notes |
|---------------------|-----------------------------------------------|--------|-------|
| `risk_return.make`  | `df_summary` columns = AnnReturn, AnnVol, TrackingErr, Agent, CVaR, MaxDD, ShortfallProb | `go.Figure` | Adds grey “sweet‑spot”, TE/ER lines & traffic‑light colours |
| `fan.make`          | `df_paths` shape = (n_sim, n_months)          | `go.Figure` | Median + confidence ribbon, optional liability overlay |
| `path_dist.make`    | same `df_paths`                               | `go.Figure` | Histogram with toggleable CDF view |
| `corr_heatmap.make` | dict of agent → df_paths                      | `go.Figure` | Monthly return ρ |
| `sharpe_ladder.make`| `df_summary`                                  | `go.Figure` | Sorted bar; hover shows ExcessReturn |
| `rolling_panel.make`| `df_paths`                                    | `go.Figure` | 3× rolling drawdown, TE and Sharpe |
| `surface.make`      | parameter grid                                | `go.Figure` | 3‑D risk/return surface |
| `grid_heatmap.make` | same parameter grid                           | `go.Figure` | 2‑D heatmap of the sweep |
| `category_pie.make` | agent → capital mapping                       | `go.Figure` | Donut by category |
| `animation.make`    | `df_paths`                                    | `go.Figure` | Animated cumulative return |
| `panel.make`        | `df_summary`                                  | `go.Figure` | Risk‑return & Sharpe ladder panel |
| `scatter_matrix.make` | any DataFrame                                | `go.Figure` | Pairwise scatter plot matrix |
| `risk_return_bubble.make` | `df_summary` with `Capital`               | `go.Figure` | Bubble-scaled risk‑return |
| `rolling_var.make`  | `df_paths`                                    | `go.Figure` | Rolling VaR line |
| `breach_calendar.make` | summary by month                           | `go.Figure` | Heatmap of TE & shortfall breaches |

*All functions must be **pure** (no I/O) and honour the colour‑blind‑safe palette defined in `viz.theme.TEMPLATE`.*

### 12.2  Streamlit app (`dashboard/app.py`)

* **Sidebar** – sliders: sims, horizon; multiselect: agents; numeric: risk‑free rate.  
* **Tabs** – Headline (risk‑return), Funding fan, Path dist, Diagnostics.  
* **Download** – `st.download_button` returns the latest PNGs and `Outputs.xlsx`.

### 12.3  CLI wrapper (`scripts/visualise.py`)

*Synopsis*  
```bash
python scripts/visualise.py \
  --plot risk_return --xlsx Outputs.xlsx \
  --agent InternalPAAgent \
  --png --pdf
Behaviour: loads Excel once → routes to viz.* → saves images under plots/.

Add if __name__ == "__main__": guard so Codex can insert into package.

12.4  Traffic‑light thresholds (central file)
config_thresholds.yaml
shortfall_green: 0.05   # ≤ 5 %
shortfall_amber: 0.10   # 5‑10 %
drawdown_limit: 0.05    # 5 % rolling 12 m
te_cap: 0.03            # 3 %
excess_return_target: 0.05  # ≥ 5 %
excess_return_floor: 0.03    # 3‑5 % amber
sharpe_green: 0.5
sharpe_amber: 0.4
confidence: 0.95

viz.theme reads this file so Ops can tweak without touching Python.

12.5  Unit tests (tests/test_viz.py)
Smoke‑test each figure: assert isinstance(fig, go.Figure) and JSON serialisable.

Snapshot regression: plotly.io.to_image(fig) hash against stored PNG bytes.

---
### 12.6  Data flow and export
1. `simulations.py` writes summary tables to pandas DataFrames.
2. `reporting/excel.py` saves those tables to `Outputs.xlsx`.
3. `scripts/visualise.py` or the dashboard load the same data and call `viz.*`.
4. Figures are returned as `go.Figure` and either displayed or exported via Plotly.

### 12.7  Styling conventions
Charts must use the template in `viz.theme.TEMPLATE`.
Fonts default to `"Roboto"` and colours follow the WCAG 2.1 contrast rules.
Avoid direct calls to `fig.update_layout` outside the theme module.

### 12.8  Extending the dashboard
New visual elements go under `viz/` and should be registered in `dashboard/app.py` via a lazy import.
Keep functions pure and document expected DataFrame columns.

### 12.9  Using `viz` in notebooks
Load the simulation outputs and display interactive charts directly in Jupyter:

```python
import pandas as pd
from pa_core.viz import risk_return, fan

df_summary = pd.read_excel("Outputs.xlsx", sheet_name="summary")
df_paths = pd.read_parquet("paths.parquet")

risk_return.make(df_summary).show()
fan.make(df_paths).show()
```

Each helper returns a `plotly.graph_objects.Figure` that renders inline via
`.show()`.

### 12.10  Batch export helper
Automate production graphics by looping over the CLI wrapper:

```bash
for chart in risk_return fan sharpe_ladder; do
    python scripts/visualise.py --plot "$chart" --xlsx Outputs.xlsx --png
done
```
This stores themed images under `plots/` ready for board packs.


### 12.11  Customising the theme
`viz.theme` centralises the Plotly template and colour mapping used across all
charts. Colours come from a colour‑blind‑safe palette and thresholds are loaded
from `config_thresholds.yaml`.

```python
from pa_core.viz import theme

# Inspect or tweak the palette
print(theme.TEMPLATE.layout.colorway)

# Map new agent classes to existing categories for consistent colours
theme.CATEGORY_BY_AGENT["OverlayOptionsAgent"] = "External Portable Alpha"
```

To tweak the traffic-light thresholds at runtime (e.g. in a notebook), call:

```python
theme.reload_thresholds("custom_thresholds.yaml")
```

Editing the YAML file or the mapping dictionary lets Ops adjust visuals without
changing any plotting code.

### 12.12  Annotation & hover tips
Use Plotly's built‑in helpers to keep charts self‑explanatory:

* `fig.add_vline` and `fig.add_hline` mark key thresholds such as the TE cap.
* Hover templates should mention the agent name and relevant metrics.

Example snippet to show CVaR on hover:

```python
hover = (
    "%{text}<br>TE=%{x:.2%}<br>ER=%{y:.2%}<br>CVaR=%{customdata:.2%}"  # noqa: W605
)
fig.add_trace(
    go.Scatter(
        x=df["AnnVol"],
        y=df["AnnReturn"],
        customdata=df["CVaR"],
        text=df["Agent"],
        hovertemplate=hover,
    )
)
```

### 12.13  Interactive HTML export
For ad‑hoc sharing, each figure can be saved as a standalone HTML file:

```python
fig.write_html("risk_return.html", include_plotlyjs="cdn")
# or use the helper
from pa_core.viz import html_export
html_export.save(fig, "risk_return.html")
```

Bundle these files under `plots/` so they drop straight into an intranet page or
email attachment.  The Streamlit dashboard links to the latest HTML exports for
offline viewing.

### 12.14  Composite layouts
Use Plotly's `make_subplots` to assemble multiple themed figures into a single
dashboard view.

```python
from plotly.subplots import make_subplots

def make_panel(df_summary, df_paths):
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(risk_return.make(df_summary).data[0], row=1, col=1)
    fig.add_trace(sharpe_ladder.make(df_summary).data[0], row=1, col=2)
    fig.update_layout(template=theme.TEMPLATE)
    return fig
```
The library exposes this helper as `viz.panel.make` for reuse in the CLI and
dashboard.

### 12.15  Caching heavy calculations
The Streamlit app may reuse the same large path matrices across widgets. Use
``@st.cache_data(ttl=600)`` so data refreshes every ten minutes without
recomputing on each interaction.

### 12.16  Animation & video export
For marketing or board presentations, animated graphics can illustrate how risk
metrics evolve over time. Plotly supports `plotly.io.write_image` with a GIF or
MP4 writer installed (e.g. `kaleido`).  Call `viz.animation.make(df_paths)` to
build an animated figure of the median cumulative return. The CLI wrapper
exposes an `--gif` flag that uses this helper and saves the resulting file
under `plots/`.

### 12.17  Real-time dashboard refresh
When simulations run continuously, Streamlit widgets can poll for new output
files on an interval.  The dashboard includes an **Auto-refresh** checkbox that
sleep-waits for the chosen interval then calls ``st.experimental_rerun()``.
Keep the data layer read-only so users cannot accidentally rerun heavy
calculations.

### 12.18  Accessibility considerations
All charts must remain readable for colour-blind users and screen readers.
Provide descriptive titles (`fig.update_layout(title_text="..."`) and add
alt-text for exported images.  Stick to the WCAG contrast ratios defined in
`viz.theme` and avoid relying solely on hue to convey meaning.

### 12.19  Rolling-metrics panel
Plot rolling drawdown, tracking error and Sharpe ratios in a single subplot
grid.  A helper `rolling_panel.make(df_paths)` computes the
metrics using a 12‑month window and returns a themed figure with three stacked
time series.  This complements the existing static metrics table and highlights
stability over time.

### 12.20  Parameter‑sweep surface
Long‑running optimisation jobs may produce a grid of results, e.g. varying
Active‑Extension leverage and external PA fraction.  Function
`surface.make(df_grid)` renders a
3‑D surface so PMs can visually pick the risk/return sweet‑spot.  The axis
labels are pulled from the DataFrame columns and the theme colours are reused
for consistency.

### 12.21  PowerPoint export
To assemble board‑pack slides directly from Python, `viz.pptx_export.save(figs,
path)` writes each Plotly figure to a slide using python‑pptx.  Combined with
the batch helper, this produces a fully branded deck without manual screenshot
hassle.

```python
figs = [risk_return.make(df_summary), fan.make(df_paths)]
pptx_export.save(figs, "board_pack.pptx")
```

### 12.22  Category-based colouring
`viz.theme.CATEGORY_BY_AGENT` maps each agent class to a descriptive
category.  The chart helpers use this mapping so that colours remain
consistent across plots.  Update the dictionary before plotting to apply a
bespoke palette or to register new agents.

```python
from pa_core.viz import theme, risk_return

theme.CATEGORY_BY_AGENT["OverlayOptionsAgent"] = "External Portable Alpha"
fig = risk_return.make(df_summary)
fig.show()
```

### 12.23  Archiving figures
Every visualisation returns a `go.Figure`.  Use Plotly's built‑in serializers
to save static SVG or the full JSON specification alongside model outputs.

```python
fig = sharpe_ladder.make(df_summary)
fig.write_image("ladder.svg")
with open("ladder.json", "w") as fh:
    fh.write(fig.to_json())
```

### 12.24  Capital-allocation donut
`viz.category_pie.make` summarises the capital invested in each agent
category as a donut chart. Pass a mapping of agent name to capital and the
function groups by `viz.theme.CATEGORY_BY_AGENT` to keep colours consistent.

```python
from pa_core.viz import category_pie

cap = {
    "BaseAgent": 500,
    "ExternalPAAgent": 300,
    "InternalPAAgent": 200,
}
fig = category_pie.make(cap)
fig.show()
```

### 12.25  Animated funding path
`viz.animation.make` builds a Plotly figure with frames showing how the median
cumulative return evolves month by month. Use the CLI flag `--gif` to export the
animation:

```bash
python scripts/visualise.py --plot fan --xlsx Outputs.xlsx --gif
```
The resulting GIF in `plots/` can be embedded in presentations or emails.

### 12.26  Path-distribution & CDF
`viz.path_dist.make` draws a histogram of end-point returns with an optional CDF toggle.

```python
from pa_core.viz import path_dist
fig = path_dist.make(df_paths)
fig.show()
```

### 12.27  Correlation heatmap
`viz.corr_heatmap.make` visualises monthly-return correlations across agents.

```python
from pa_core.viz import corr_heatmap
fig = corr_heatmap.make({"PA": df_paths_pa, "AE": df_paths_ae})
fig.show()
```

### 12.28  Sharpe-ratio ladder
`viz.sharpe_ladder.make` ranks agents by Sharpe ratio in a single bar chart.

```python
from pa_core.viz import sharpe_ladder
fig = sharpe_ladder.make(df_summary)
fig.show()
```

### 12.29  Risk–return scatter
`viz.risk_return.make` plots tracking error on the x-axis and excess return on the y-axis. Points are coloured by shortfall probability and the sweet-spot box is derived from `config_thresholds.yaml`.

```python
fig = risk_return.make(df_summary)
fig.show()
```



### 12.30  Multi-agent overlay
`viz.overlay.make` plots the median cumulative return of several agents on a single chart for quick comparisons.
Colours follow `viz.theme.CATEGORY_BY_AGENT` so the same agent categories share a consistent hue across all charts.
The palette is drawn from `viz.theme.TEMPLATE`; if that template lacks a
`colorway`, Plotly's default colours are used as a fallback.

```python
from pa_core.viz import overlay
fig = overlay.make({"Base": df_paths_base, "AE": df_paths_ae})
fig.show()
```

### 12.31  Risk-contribution waterfall
Use `viz.waterfall.make` to visualise how each sleeve contributes to total tracking error or expected return.

```python
from pa_core.viz import waterfall
contrib = {"InternalPAAgent": 0.08, "ExternalPAAgent": 0.02}
fig = waterfall.make(contrib)
fig.show()
```

### 12.32  Scenario slider animation
`viz.scenario_slider.make` builds a figure with a slider to step through precomputed Plotly frames. Combine with `viz.animation.make` to present stress tests interactively.
```python
frames = [
    go.Frame(data=[go.Scatter(y=arr[i].cumsum())], name=str(i))
    for i in range(5)
]
fig = scenario_slider.make(frames)
fig.show()
```

### 12.33  Export bundle helper

Call `viz.export_bundle.save(figs, "plots/summary")` to output PNG, HTML and JSON
files for each figure in one go.  This is handy for archiving runs or sharing via
email.

### 12.34  Chart gallery notebook
A Jupyter notebook `viz_gallery.ipynb` will demonstrate each function with
sample data. PMs can tweak parameters live to see how colours and thresholds
respond. The gallery acts as living documentation for Ops and quants
experimenting with new sleeves.

### 12.35  Data tables and grids
A helper `viz.data_table.make(df)` will render a sortable
`dash_table.DataTable` using the same colour rules as the charts. Tables
summarise path metrics and can be downloaded as CSV via a built-in button.

### 12.36  Combined scenario viewer
`viz.scenario_viewer.make` ties together the slider animation and overlay chart
into a single figure with tabs. This lets users flip between cumulative paths
and funding-level trajectories without losing context.

### 12.37  PDF export convenience
The CLI's `--pdf` flag will call `fig.write_image("name.pdf")` for every figure
in the export bundle. Use this when board packs require static attachments
rather than interactive HTML.

### 12.38  Scheduled snapshot archiver
An optional cron job can call `export_bundle.save` nightly so the dashboard
always links to the latest set of figures. Archived images are stamped with the
run timestamp to aid traceability.

### 12.39  Future ideas
Potential extensions include violin plots of monthly returns, live WebSocket
hooks for streaming results, and integration with Bokeh for alternate themes.
Feel free to propose additional visualisations as needs evolve.

### 12.40  Parameter-grid heatmap
`viz.grid_heatmap.make` complements the 3‑D surface with a top-down view of the
same data. This is useful when the sweep only spans a handful of values.

```python
grid = pd.read_csv("sweep.csv")
fig = grid_heatmap.make(grid)
fig.show()
```

### 12.41  Violin distribution of returns
`viz.violin.make` plots the distribution of monthly returns either aggregated
across all months or one violin per month when `by_month=True`.

```python
from pa_core.viz import violin
fig = violin.make(df_paths, by_month=True)
fig.show()
```

### 12.42  Rolling correlation heatmap
`viz.rolling_corr_heatmap.make` visualises how correlations evolve over time
with a sliding window.  Provide a `(n_sim, n_months)` array of monthly returns
and choose the window length in months.  The helper computes the correlation
between each month and its predecessors up to the chosen lag so users can spot
periods of heightened autocorrelation.

```python
from pa_core.viz import rolling_corr_heatmap
fig = rolling_corr_heatmap.make(df_paths, window=12)
fig.show()
```

### 12.43  Exposure timeline
`viz.exposure_timeline.make` plots each agent's capital allocation as a stacked
area chart so PMs can verify exposure drift across the horizon.  The input is a
DataFrame indexed by month with one column per agent.  Values should represent
capital in the sleeve's base currency.

```python
from pa_core.viz import exposure_timeline
fig = exposure_timeline.make(capital_by_month)
fig.show()
```

### 12.44  Risk-target gauge
Display tracking error or CVaR relative to thresholds using
`viz.gauge.make(df_summary)`.  Pass a summary table with the metric of interest
(``TrackingErr`` or ``CVaR``).  The dial turns amber or red when the value
breaches the levels in ``config_thresholds.yaml``.

```python
from pa_core.viz import gauge
fig = gauge.make(df_summary)
fig.show()
```

### 12.45  Parameter-sensitivity radar
Compare multiple scenarios on a single radar chart with
`viz.radar.make(df_metrics)`.  Provide a DataFrame whose rows are scenarios and
whose columns are metrics such as TE and ER.  Each trace plots the metrics in a
closed loop so sensitivities leap out visually.

```python
from pa_core.viz import radar
fig = radar.make(df_metrics)
fig.show()
```

### 12.46  Scatter-matrix diagnostic
`viz.scatter_matrix.make` draws a grid of pairwise scatter plots across a set of
metrics so relationships between them become obvious.  Pass a DataFrame where
columns are metrics such as TE, ER and CVaR.  The helper shades points by agent
category using `viz.theme.CATEGORY_BY_AGENT`.

```python
from pa_core.viz import scatter_matrix
fig = scatter_matrix.make(df_summary[["TrackingErr", "AnnReturn", "CVaR"]])
fig.show()
```

### 12.47  Weighted risk‑return bubbles
When the capital invested in each agent differs materially, use
`viz.risk_return_bubble.make` to scale each point by investment weight.
Inputs match `risk_return.make` with an additional `Capital` column.

### 12.48  Rolling VaR timeline
`viz.rolling_var.make(df_paths, window=12)` plots the rolling Value at Risk or
CVaR over the chosen horizon so breaches stand out visually.

### 12.49  Breach-calendar heatmap
Highlight months where tracking error or shortfall exceeded thresholds with
`viz.breach_calendar.make(df_summary)`.  The output is a Plotly heatmap
compatible with the Streamlit dashboard.

### 12.50  Scenario comparison table
`viz.data_table.make(df)` already powers sortable tables.  Combining it with
`viz.export_bundle.save` produces an HTML bundle containing the figures and the
table so PMs can explore results offline.

### 12.51  Rolling-moments panel
`viz.moments_panel.make` charts rolling skewness and kurtosis so heavy tails are
easy to spot. Pass the same `df_paths` matrix used by other helpers and adjust
the `window` argument to change the lookback period.

```python
from pa_core.viz import moments_panel
fig = moments_panel.make(df_paths, window=12)
fig.show()
```

### 12.52  Parallel-coordinates view
Compare multiple metrics simultaneously with `viz.parallel_coords.make(df)`. The
input DataFrame's columns become the parallel axes, allowing quick detection of
outliers across dimensions.

```python
from pa_core.viz import parallel_coords
fig = parallel_coords.make(df_summary[["AnnReturn", "AnnVol", "TrackingErr"]])
fig.show()
```

### **13  CLI Additions** &nbsp;*(new subsection in cli.py docstring)*

// NEW  
```text
--png / --pdf / --pptx     Static exports (can be combined)
--html                    Save interactive HTML
--gif                     Animated export of monthly paths
--dashboard                Launch Streamlit dashboard after run

```
