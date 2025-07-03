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
| Chart functions vectorised | 🚫 none | Implement in viz/ pure‑numpy layers |


7  Reporting
// NEW

Excel – reporting/excel.py now calls viz.risk_return.make() etc. and embeds the PNG exports as worksheet objects (use xlsxwriter.Workbook.insert_image).

Static exports – scripts/visualise.py accepts --png --pdf --pptx; each flag triggers fig.write_image() or python‑pptx helper.

Dashboard – dashboard/app.py launches automatically when --launch_dashboard flag is set or via streamlit run dashboard/app.py.


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
| `risk_return.make`  | `df_summary` columns = AnnReturn, AnnVol, TrackingErr, Agent, CVaR, MaxDD, ShortfallProb | `go.Figure` | Adds grey “sweet‑spot” rectangle & traffic‑light colours |
| `fan.make`          | `df_paths` shape = (n_sim, n_months)          | `go.Figure` | Plots median + 95 % ribbon vs. liability glide‑path |
| `path_dist.make`    | same `df_paths`                               | `go.Figure` | Histogram and CDF slider |
| `corr_heatmap.make` | dict of agent → df_paths                      | `go.Figure` | Monthly return ρ |
| `sharpe_ladder.make`| `df_summary`                                  | `go.Figure` | Sorted bar; hover shows ExcessReturn |

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

### 12.15  Caching heavy calculations
The Streamlit app may reuse the same large path matrices across widgets. Use
`@st.cache_data` with an explicit TTL to prevent repeated computations while
keeping memory use in check.

### 12.16  Animation & video export
For marketing or board presentations, animated graphics can illustrate how risk
metrics evolve over time. Plotly supports `plotly.io.write_image` with a GIF or
MP4 writer installed (e.g. `kaleido`).  The CLI wrapper exposes an `--gif`
flag that iterates over months and saves the resulting animation under
`plots/`.

### 12.17  Real-time dashboard refresh
When simulations run continuously, Streamlit widgets can poll for new output
files on an interval.  Use `st.experimental_rerun()` inside a timer thread to
reload figures without manual refresh.  Keep the data layer read-only so users
cannot accidentally rerun heavy calculations.

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


### **13  CLI Additions** &nbsp;*(new subsection in cli.py docstring)*

// NEW  
```text
--png / --pdf / --pptx     Static exports (can be combined)
--dashboard                Launch Streamlit dashboard after run

```
