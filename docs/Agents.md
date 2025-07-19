# Tutorial Enhancement Guide
*Portable‑Alpha + Active‑Extension Model*

> "Make everything as simple as possible, but no simpler." – A. Einstein  

---

## � **TESTING DOCUMENTATION & REFERENCES**

### **Complete Testing Results**
- **`TUTORIAL_5_TESTING_RESULTS.md`**: Comprehensive visualization testing with dependency analysis and user experience findings
- **`STREAMLIT_TUTORIAL_INTEGRATION_PLAN.md`**: Detailed Streamlit integration plan for enhanced tutorial accessibility

### **Implementation Status**
- ✅ **All 10 tutorials tested** with complete documentation of functionality and issues
- ✅ **Parameter sweep engine** operational (3 of 4 modes working)
- ✅ **Visualization system** complete with export bundle functionality
- 🔶 **Tutorial enhancement implementation** ready following priority order below

---

## �📚 TUTORIAL ENHANCEMENT PRIORITIES

**Last Updated:** July 15, 2025  
**Status:** All 10 tutorials tested, comprehensive enhancement plans complete

### 🎯 HIGH PRIORITY TUTORIAL ENHANCEMENTS

**Ready for Implementation**: Parameter sweep engine working for 3 of 4 modes enables comprehensive tutorial upgrades from basic single-scenario demonstrations to professional multi-scenario analysis workflows.

#### 🎓 **Tutorial 1: Multi-Mode Parameter Sweep Demonstration (HIGH PRIORITY)**
**Current**: Basic single-scenario introduction  
**Critical Issues**: None - foundational tutorial works correctly  
**Enhancement**: 5-part comprehensive tutorial showcasing parameter sweep capabilities

**Implementation Plan**:
- ✅ **Part 1**: Basic Operation - Use `config/params_template.yml` for single scenario foundation
- ✅ **Part 2**: Capital Allocation Optimization - Use `config/capital_mode_template.csv` with `--mode capital` (38KB Excel output, working)
- ✅ **Part 3**: Alpha Capture Efficiency - Use `config/alpha_shares_mode_template.csv` with `--mode alpha_shares` (183KB Excel, 187 combinations, working)  
- ✅ **Part 4**: Volatility Stress Testing - Use `config/vol_mult_mode_template.csv` with `--mode vol_mult` (13KB Excel output, working)
 - ✅ **Part 5**: Return Sensitivity Analysis - Use `config/returns_mode_template.csv` with `--mode returns` (bug fixed)

#### 🎓 **Tutorial 2: Advanced Threshold Analysis (MEDIUM PRIORITY)**
**Current**: Basic metric interpretation  
**Critical Issues**: 
- ❌ **HIGH**: 3 of 4 agents exceed TE budget (10% vs 3% cap) - needs interpretation guidance
- ⚠️ **MEDIUM**: Missing context on normal vs concerning results, no actionable steps when thresholds breached

**Enhancement**: Advanced multi-scenario analysis using parameter sweep outputs
- Bulk scenario threshold analysis (50-200 scenarios at once)
- Multi-sheet Excel interpretation techniques for large-scale parameter exploration
- Systematic workflows to identify threshold-compliant parameter combinations

#### 🎓 **Tutorial 3: Multi-Scenario Dashboard Workflows (MEDIUM PRIORITY)**
**Current**: Basic dashboard and visualization tutorial  
**Critical Issues**:
- ❌ **HIGH**: Streamlit not pre-installed - blocks tutorial completely for new users
- ❌ **HIGH**: Chrome dependency for PNG/PDF exports - image generation fails
- ⚠️ **MEDIUM**: AllReturns sheet context missing, dashboard navigation guidance insufficient

**Enhancement**: Multi-scenario visualization workflows using parameter sweeps
- Multi-scenario dashboard loading (parameter sweep Excel files: 38KB-183KB outputs)
- Bulk visualization workflows (compare 50-200 scenarios simultaneously)
- Parameter sensitivity and threshold compliance visualization techniques

#### 🎓 **Tutorial 4: Professional Bulk Export Workflows (MEDIUM PRIORITY)**
**Current**: Basic chart export tutorial  
**Critical Issues**:
- ✅ **WORKING**: HTML, PDF, PPTX exports work correctly with alt-text support
- ⚠️ **MEDIUM**: PNG export dependency issues, limited format guidance

**Enhancement**: Professional bulk export workflows using parameter sweeps
- Generate presentation decks from 50-200 scenarios automatically
- Multi-scenario chart workflows and client-ready reporting automation
- Integrate with parameter sweep results for systematic documentation

#### 🎓 **Tutorial 5: Automated Bulk Visualization Workflows (MEDIUM PRIORITY)**
**Current**: Basic script-based visualizations  
**Critical Issues** (documented in `TUTORIAL_5_TESTING_RESULTS.md`):
- ✅ **HTML exports**: Interactive charts work for risk_return, corr_heatmap, sharpe_ladder, rolling_panel, surface
- ❌ **HIGH**: Chrome/Kaleido dependency for static exports - PNG/PDF/PPTX fail silently without Chrome installed
- ❌ **MEDIUM**: Parquet file requirement for fan/path_dist plots - basic simulations don't generate required `Outputs.parquet`
- ⚠️ **MEDIUM**: GIF export logging and feedback - failures only appear in logs, users don't see warnings
- ⚠️ **MEDIUM**: Plot type descriptions missing - tutorial lists plot names without use case guidance

**Enhancement**: Automated bulk visualization workflows using parameter sweeps
- Bulk chart generation over parameter sweep results (50-200 plots)
- Multi-scenario gallery creation and integrated client reporting workflows
- Professional visualization pipelines for systematic analysis
- **Implementation Priority**: Fix dependency issues first, then enhance with parameter sweep integration

#### 🎓 **Tutorial 6: Dynamic Agent Configuration (LOW PRIORITY)**
**Current**: Custom agent creation  
**Critical Issues**:
- ❌ **HIGH**: ModelConfig schema rigid; cannot allocate capital to custom agents via config
- ⚠️ **MEDIUM**: Tutorial lacks example code template, no guidance to extend ModelConfig/build_from_config

**Enhancement**: Enable dynamic agent configuration and streamlined custom agent integration
- Bulk performance sweeps for custom agents across capital allocation ranges
- Integration of custom agent metrics into sweep results and visualizations
- Streamlined registration and testing workflows for new agent types

#### 🎓 **Tutorial 7: Advanced Theme Integration (LOW PRIORITY)**
**Current**: Theme and threshold customization  
**Critical Issues**: None - reload functions validated and working  
**Minor Gaps**: Tutorial text lacks example code snippets, no integration with parameter sweep reports demonstrated

**Enhancement**: Best-practice snippets and custom theme integration with parameter sweeps
- Scenario-specific styling per sweep scenario
- Threshold-driven color mapping for sweep visualizations
- Automated theme application for batch exports and professional reporting

#### 🎓 **Tutorial 8: Automated Stress Testing (MEDIUM PRIORITY)**
**Current**: Manual scenario stress-testing  
**Critical Issues**:
- ❌ **HIGH**: CLI overwrites default output without `--output` guidance
- ⚠️ **MEDIUM**: Manual Python/Excel comparison burdens users, dashboard lacks multi-scenario comparison

**Enhancement**: Automate stress tests using parameter sweep engine and built-in comparison tools
- Automated scenario sweeps across capital, alpha, and financing parameters
- Built-in diff reports and dashboard multi-scenario mode for side-by-side visualizations
- Replace manual stress testing with systematic parameter sweep workflows

#### 🎓 **Tutorial 9: Enhanced Export Bundle Integration (LOW PRIORITY)**
**Current**: Basic `export_bundle.save` helper  
**Critical Issues**:
- ✅ **WORKING**: PNG, HTML, JSON outputs generated successfully with alt-text support
- ⚠️ **MEDIUM**: Static export failures now log warnings but need better user feedback
- ⚠️ **MEDIUM**: Limited format support (no PDF, PPTX, GIF by default), fixed sequential naming

**Enhancement**: Extend export bundle for more formats, error logging, custom naming, CLI/sweep integration
- Bulk export of all chart variants across sweep results
- Plugin-based additional formats and scenario-labeled file stems for traceable output archives
- Integration with parameter sweep workflows for automated documentation

#### 🎓 **Tutorial 10: Interactive Gallery Enhancement (LOW PRIORITY)**
**Current**: `viz_gallery.ipynb` demonstrations  
**Critical Issues**:
- ⚠️ **MEDIUM**: No built-in data sources (manual dataframe creation required)
- ⚠️ **MEDIUM**: Lacks file export examples, minimal markdown guidance, no batch gallery examples

**Enhancement**: Enhanced notebook with real simulation data imports, export demos, narrative guidance
- In-notebook batch export workflows using `export_bundle` or `visualise.py`
- Markdown commentary for each chart type and theme/threshold reload examples
- Integration with parameter sweep results for comprehensive gallery demonstrations

#### 📊 **Implementation Priority Order**

**PHASE 1: Core Tutorial Enhancements**
1. **HIGH**: Tutorial 1 enhancement (all five parts working including returns mode)
2. **MEDIUM**: Tutorial 2 enhancement using sweep outputs (threshold analysis workflows)
3. **MEDIUM**: Tutorial 3 enhancement after dependency fixes (multi-scenario dashboard workflows)
4. **MEDIUM**: Tutorial 4 enhancement using sweep outputs (bulk export and reporting workflows)
5. **MEDIUM**: Tutorial 5 enhancement using sweep outputs (automated visualization workflows)
6. **MEDIUM**: Tutorial 8 enhancement (replace manual stress testing with automated sweeps)
7. **LOW**: Tutorial 6 enhancement (custom agent configuration and integration)
8. **LOW**: Tutorial 7 enhancement (advanced theme integration with sweeps)
9. **LOW**: Tutorial 9 enhancement (export bundle extensions and CLI integration)
10. **LOW**: Tutorial 10 enhancement (interactive gallery and comprehensive documentation)

**PHASE 2: Advanced User Experience** *(Start only after Phase 1 complete)*
11. **FUTURE**: **Streamlit Tutorial Integration** - Implement comprehensive plan from `STREAMLIT_TUTORIAL_INTEGRATION_PLAN.md`
    - Guided tutorial mode within Streamlit app (eliminates CLI barriers)
    - Interactive parameter configuration and validation
    - Real-time tutorial progress tracking and state management
    - Integrated visualization generation and export workflows
    - **Prerequisites**: All core tutorial enhancements complete, dependency issues resolved

### 🔧 **Critical Bug Fixes Required**
- **Returns Mode CLI Bug**: **Fixed** – returns sweep now works via `--mode returns`
- **Streamlit Installation**: Add Streamlit to default installation instructions
- **Chrome/Kaleido Dependencies**: Document static export requirements and provide installation guidance

### 💡 **Key Enhancement Themes**
1. **Parameter Sweep Integration**: Transform all tutorials from single-scenario to comprehensive multi-scenario analysis
2. **Professional Reporting**: Enable bulk export, automated documentation, and client-ready presentations
3. **Advanced Analytics**: Systematic threshold analysis, sensitivity testing, and optimization workflows
4. **User Experience**: Reduce manual steps, provide clear guidance, improve error feedback
5. **Scalability**: Support analysis of 50-200 scenarios simultaneously across all tutorial workflows
6. **Future: Streamlit Integration**: Comprehensive tutorial integration within Streamlit app (detailed plan in `STREAMLIT_TUTORIAL_INTEGRATION_PLAN.md`) - *implement after core enhancements complete*

---

## 🎯 **DEVELOPMENT COORDINATION**

### ✅ **Completed Work (Do Not Modify)**
- **Parameter Sweep Engine**: ✅ All 4 analysis modes implemented and tested
- **Core Agent Infrastructure**: ✅ All agents implemented and validated
- **CLI and Dashboard**: ✅ Core functionality working correctly
- **Visualization Architecture**: ✅ Complete chart library with theme system
- **Configuration System**: ✅ ModelConfig validation and template system

### ⚠️ **Work In Progress**
- **Tutorial Enhancement Implementation**: Following priority order above
- **Documentation Updates**: Integrating parameter sweep capabilities into user guides
- **Dependency Management**: Streamlining installation and setup processes

### 🤝 **Coordination Guidelines**
- Use feature branches for tutorial enhancements: `feature/tutorial-N-enhancement`
- Test all parameter sweep modes before implementing tutorial changes
- Maintain backward compatibility with existing single-scenario workflows
- Document new multi-scenario capabilities thoroughly
- Run full test suite before merging tutorial enhancement work

---

## 📚 **TUTORIAL-SPECIFIC IMPLEMENTATION DETAILS**

The following sections have been removed as they contain completed architectural work:
- Agent interface specifications (implemented and working)
- Package layout details (structure finalized)
- Simulation engine specifications (vectorization complete)
- Configuration system details (ModelConfig implemented)
- Visualization architecture (complete chart library available)

**Focus**: All development effort should concentrate on the tutorial enhancement plans outlined above, leveraging the completed parameter sweep engine to transform basic single-scenario tutorials into comprehensive multi-scenario analysis workflows.

---

## 📋 **QUICK REFERENCE FOR TUTORIAL ENHANCEMENT WORK**

### 🚀 **Getting Started with Tutorial Enhancement**

**Step 1**: Choose a tutorial from the priority list above  
**Step 2**: Create feature branch: `git checkout -b feature/tutorial-N-enhancement`  
**Step 3**: Test relevant parameter sweep modes before implementation  
**Step 4**: Implement enhancements following the detailed plans above  
**Step 5**: Document new multi-scenario capabilities  
**Step 6**: Run full test suite and submit PR  

### 🔧 **Critical Files for Tutorial Enhancement**
- **Tutorial Source**: `docs/UserGuide.md` (tutorials 1-10)
- **Parameter Templates**: `config/` directory (sweep mode templates)
- **Test Results**: Tutorial testing documentation files
- **CLI Interface**: `pa_core/cli.py` (returns mode sweep fixed)
- **Dashboard**: `dashboard/app.py` (Streamlit integration)
- **Visualization**: `scripts/visualise.py` (bulk export workflows)

### 📊 **Parameter Sweep Integration Points**
- **Tutorial 1**: Multi-mode demonstration using all 4 sweep modes
- **Tutorials 2-5, 8**: Advanced workflows using sweep outputs (50-200 scenarios)
- **Tutorials 6-7, 9-10**: Integration enhancements and advanced features

### 🎯 **Success Metrics for Tutorial Enhancement**
- ✅ **Functionality**: All parameter sweep modes working in tutorial context
- ✅ **Documentation**: Clear step-by-step instructions for multi-scenario analysis
- ✅ **User Experience**: Reduced manual steps, improved error feedback
- ✅ **Scalability**: Support for 50-200 scenario analysis workflows
- ✅ **Professional Output**: Client-ready reporting and visualization capabilities

---

*This guide focuses exclusively on tutorial enhancement implementation. All architectural work has been completed and should not be modified. Direct all development effort toward transforming the existing 10 tutorials into comprehensive parameter sweep demonstrations.*

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

Example values are provided in `config/params_template.yml` with the same
fields mirrored for spreadsheet users in `config/parameters_template.csv`.
Each key corresponds to an attribute on the `ModelConfig` dataclass.

| Key | Description |
| --- | ----------- |
| `N_SIMULATIONS` | Number of Monte Carlo trials |
| `N_MONTHS` | Months per simulation run |
| `external_pa_capital` | Capital allocated to external PA sleeve ($ mm) |
| `active_ext_capital` | Capital allocated to active extension sleeve ($ mm) |
| `internal_pa_capital` | Capital allocated to internal portable alpha ($ mm) |
| `total_fund_capital` | Total fund size ($ mm) |
| `w_beta_H` | Beta weight in the internal sleeve |
| `w_alpha_H` | Alpha weight in the internal sleeve |
| `theta_extpa` | Fraction of external PA capital used for alpha |
| `active_share` | Active‑extension active share fraction |
| `mu_H` | Annual mean return of in‑house alpha |
| `sigma_H` | Annual volatility of in‑house alpha |
| `mu_E` | Annual mean return of extension alpha |
| `sigma_E` | Annual volatility of extension alpha |
| `mu_M` | Annual mean return of external PA alpha |
| `sigma_M` | Annual volatility of external PA alpha |
| `rho_idx_H` | Correlation of index vs in‑house alpha |
| `rho_idx_E` | Correlation of index vs extension alpha |
| `rho_idx_M` | Correlation of index vs external PA alpha |
| `rho_H_E` | Correlation of in‑house vs extension alpha |
| `rho_H_M` | Correlation of in‑house vs external PA alpha |
| `rho_E_M` | Correlation of extension vs external PA alpha |
| `internal_financing_mean_month` | Mean monthly financing cost for internal sleeve |
| `internal_financing_sigma_month` | Volatility of internal financing cost |
| `internal_spike_prob` | Probability of a financing spike internally |
| `internal_spike_factor` | Size multiplier for internal financing spikes |
| `ext_pa_financing_mean_month` | Mean monthly financing cost for external PA |
| `ext_pa_financing_sigma_month` | Volatility of external PA financing cost |
| `ext_pa_spike_prob` | Probability of a financing spike in external PA |
| `ext_pa_spike_factor` | Size multiplier for external PA financing spikes |
| `act_ext_financing_mean_month` | Mean monthly financing cost for active extension |
| `act_ext_financing_sigma_month` | Volatility of active extension financing cost |
| `act_ext_spike_prob` | Probability of a financing spike for active extension |
| `act_ext_spike_factor` | Size multiplier for active extension financing spikes |

// NEW –‑export & dashboard flags
@dataclass
class RunFlags:
    save_xlsx: str | None = "Outputs.xlsx"
    png: bool = False
    pdf: bool = False
    pptx: bool = False
    html: bool = False
    gif: bool = False
    dashboard: bool = False   # streamlit run after sim
    alt_text: str | None = None



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

Dashboard – `dashboard/app.py` caches loaded data with ``@st.cache_data`` and offers an optional auto‑refresh checkbox. Launch with ``--dashboard`` or run ``streamlit run dashboard/app.py``.


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

1. **Parameter defaults** – defaults are now locked in `config.py` (e.g. `rho_E_M = 0.0`).
2. **Financing spikes per sleeve** – differentiated via `internal_spike_prob`, `ext_pa_spike_prob` and `act_ext_spike_prob` fields.
3. **Random‑seed strategy** – `spawn_agent_rngs` creates deterministic per‑agent generators when a seed is supplied.
4. **Outputs.xlsx layout** – existing sheet order retained; pass `--pivot` for a long-format sheet.
5. **Dashboard theme** – ✅ corporate palette and fonts locked in `config_theme.yaml`.
6. **GUI alt text support** – `viz.html_export.save` and `viz.pptx_export.save` accept an `alt_text` parameter. The CLI exposes `--alt-text` so exported charts remain accessible.
7. **ShortfallProb metric** – `risk_metrics` must include this column. The parameter templates and config validation enforce it, while Excel and the risk‑return chart add `ShortfallProb` if missing.
8. **Animated & interactive exports** – pass `--gif` for a looping animation or `--html` for a standalone interactive page.


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
python -m pa_core.cli \
  --params parameters.csv \
  --index sp500tr_fred_divyield.csv \
  --mode returns \
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

### 12.1.1 Parameter Sweep Visualization Integration

**✅ ARCHITECTURAL DECISION: Single Visualization Suite for All Analysis Modes**

The parameter sweep engine (4 modes: capital/returns/alpha_shares/vol_mult) **reuses the existing visualization architecture** without requiring new chart types:

**Integration Pattern:**
- **All Sweep Modes** → Generate standard `df_summary` DataFrames
- **Excel Export** → Uses `viz.risk_return.make()` to embed charts automatically  
- **3D Analysis** → Uses existing `viz.surface.make()` and `viz.surface_animation.make()`
- **Dashboard** → All existing tabs work with sweep results (no modifications needed)

**Visualization Compatibility Matrix:**
| Analysis Mode | `risk_return` | `surface` | `fan` | `sharpe_ladder` | `rolling_panel` | 
|---------------|---------------|-----------|-------|-----------------|-----------------|
| Capital       | ✅ Auto       | ✅ Yes    | ✅ Yes| ✅ Yes          | ✅ Yes          |
| Returns       | ✅ Auto       | ✅ Yes    | ✅ Yes| ✅ Yes          | ✅ Yes          |
| Alpha Shares  | ✅ Auto       | ✅ Yes    | ✅ Yes| ✅ Yes          | ✅ Yes          |
| Vol Mult      | ✅ Auto       | ✅ Yes    | ✅ Yes| ✅ Yes          | ✅ Yes          |

**Key Benefits:**
- ✅ **No visualization duplication** - Single codebase serves all modes
- ✅ **Consistent user experience** - Same charts across all analysis types  
- ✅ **Maintainable architecture** - Changes to visualizations apply to all modes
- ✅ **Future-proof design** - New analysis modes automatically work with existing charts

**Implementation:**
```python
# All modes produce standard summary DataFrames that work with all viz functions
results = run_parameter_sweep(cfg, idx_series, rng_returns, fin_rngs)
# Excel export automatically includes risk-return chart
export_sweep_results(results, filename="Sweep.xlsx")  # Uses viz.risk_return.make()
```

### 12.2  Streamlit app (`dashboard/app.py`)

* **Sidebar** – sliders: sims, horizon; multiselect: agents; numeric: risk‑free rate. Theme YAML path allows on-the-fly rebranding.
* **Tabs** – Headline (risk‑return), Funding fan, Path dist, Diagnostics.
* **Download** – `st.download_button` returns the latest PNGs and `Outputs.xlsx`.
* **New tabs** – Edit the `PLOTS` mapping in `dashboard/app.py` to register
  additional `viz` functions. Each key becomes a tab label and the value is the
  dotted path to the `make` function that builds a `go.Figure`.

### 12.3  CLI wrapper (`scripts/visualise.py`)

*Synopsis*  
```bash
python scripts/visualise.py \
  --plot risk_return --xlsx Outputs.xlsx \
  --agent InternalPAAgent \
  --png --pdf --alt-text "Risk-return chart"
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
charts. Colours and fonts now load from `config_theme.yaml` so the dashboard
matches corporate branding. The default palette uses navy, blue, green, orange,
purple and red (`#003366`, `#0070c0`, `#00b050`, `#ff9900`, `#7030a0`,
`#ff0000`) with the "Source Sans Pro" font on a light grey background. Traffic
light thresholds are still loaded from `config_thresholds.yaml`.

```python
from pa_core.viz import theme

# Inspect or tweak the palette
print(theme.TEMPLATE.layout.colorway)

# Background colours are also loaded from ``config_theme.yaml``
print(theme.TEMPLATE.layout.paper_bgcolor)

# Map new agent classes to existing categories for consistent colours
theme.CATEGORY_BY_AGENT["OverlayOptionsAgent"] = "External Portable Alpha"
```

To tweak the traffic-light thresholds at runtime (e.g. in a notebook), call:

```python
theme.reload_thresholds("custom_thresholds.yaml")

# Reload colours or fonts on the fly
theme.reload_theme("my_theme.yaml")
```

Editing the YAML file or the mapping dictionary lets Ops adjust visuals without
changing any plotting code. ``config_theme.yaml`` now supports ``paper_bgcolor``
and ``plot_bgcolor`` keys so the GUI background matches corporate colours.

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
alt-text for exported images.  The ``viz.html_export.save`` and
``viz.pptx_export.save`` helpers accept an ``alt_text`` argument, and the
CLI wrapper exposes ``--alt-text`` for convenience so
screen readers can describe the figures. Stick to the WCAG contrast ratios
defined in ``viz.theme`` and avoid relying solely on hue to convey meaning.

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
A Jupyter notebook `viz_gallery.ipynb` under the project root now demonstrates
each function with sample data. PMs can tweak parameters live to see how
colours and thresholds respond.  Launch it after running `pip install -e .` to
explore the charts interactively.  The gallery acts as living documentation for
Ops and quants experimenting with new sleeves.

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

### 12.53  Capital treemap
`viz.capital_treemap.make` visualises the hierarchical distribution of capital
across agent categories using a Plotly treemap.  Pass a mapping of agent names
to capital amounts; the helper aggregates by
`viz.theme.CATEGORY_BY_AGENT` when no category mapping is provided.

```python
from pa_core.viz import capital_treemap
cap = {"BaseAgent": 500, "ExternalPAAgent": 300, "InternalPAAgent": 200}
fig = capital_treemap.make(cap)
fig.show()
```

### 12.54  Correlation network
`viz.corr_network.make` draws a network graph where edges connect agents with
monthly-return correlations above a chosen threshold.  This highlights clusters
of closely related behaviour.

```python
from pa_core.viz import corr_network
fig = corr_network.make({"A": df_a, "B": df_b}, threshold=0.3)
fig.show()
```

### 12.55  Beta-exposure heatmap
Visualise how beta exposure drifts over time with `viz.beta_heatmap.make`.
Provide a DataFrame indexed by month and columns per agent representing the
beta value for that period.

```python
from pa_core.viz import beta_heatmap
fig = beta_heatmap.make(beta_by_month)
fig.show()
```

### 12.56  Factor-exposure bar chart
Display each sleeve's exposure to common factors (e.g. Value, Momentum) with
`viz.factor_bar.make`. Pass a DataFrame whose **rows** are agent names and
**columns** are factor labels. Colours are assigned via
`viz.theme.CATEGORY_BY_AGENT` so that exposures line up visually with the other
charts. Set ``barmode="group"`` by default so each factor is easy to compare
across agents.  The helper returns a standard ``go.Figure`` which callers can
further customise (e.g. switch to ``stack`` mode) if required.

```python
from pa_core.viz import factor_bar
fig = factor_bar.make(exposures_df)
fig.show()
```

### 12.57  Multi-horizon VaR fan
`viz.multi_fan.make(df_paths, horizons=[12, 24, 36])` overlays several
confidence ribbons so PMs can compare 1‑, 2‑ and 3‑year shortfall risks in a
single plot. Each horizon is drawn in a different colour with the median path on
top of a shaded region representing the configured confidence level. Pass a
sequence of month counts to ``horizons`` to change which ribbons are included.

### 12.58  TE vs. Beta scatter
Visualise the relationship between tracking error and beta exposure using
`viz.beta_scatter.make(df_summary)`. Points are coloured by shortfall
probability (green/amber/red) and sized by the ``Capital`` column so more
material sleeves stand out. Hover text shows the agent name and numeric values
for quick inspection.

### 12.59  Weighted overlay
`viz.overlay_weighted.make` extends the basic overlay by weighting each path by
the invested capital. Pass a mapping ``{"Agent": (paths, capital)}``. Line width
scales with capital and a dashed trace shows the capital-weighted median across
all agents. This highlights which sleeve dominates the cumulative return
profile.

### 12.60  Factor-sensitivity matrix
`viz.factor_matrix.make(df)` plots a heatmap of factor sensitivities across
agents. The input DataFrame should have factors as the index and agent names as
columns. Values are visualised using ``go.Heatmap`` with the theme palette so
risk contributions are easy to compare and export alongside the numeric table.

### 12.61  TE vs. CVaR scatter
`viz.te_cvar_scatter.make(df_summary)` compares tracking error on the x-axis
with CVaR on the y-axis. Marker colour follows the shortfall-probability rules
and size scales with the ``Capital`` column when present. Hover text lists the
agent name and values so outliers pop out immediately.

### 12.62  Custom quantile fan
Use `viz.quantile_fan.make(df_paths, quantiles=(0.1, 0.9))` when different
confidence bounds are required. The helper behaves like `viz.fan.make` but
lets callers set arbitrary lower and upper quantiles. This is handy for stress
tests where 80 % or 99 % intervals are desired.

### 12.63  Return-attribution sunburst
`viz.sunburst.make(df_returns)` visualises the contribution of each sleeve and sub-strategy to the overall portfolio return using a nested sunburst chart. This helps PMs trace where performance came from at a glance.

### 12.64  Horizon slicer
`viz.horizon_slicer.make(df_paths)` adds a Plotly range slider so users can
focus on a subset of months while preserving the themed styling. Use this when
analysing long simulations in Jupyter:

```python
from pa_core.viz import horizon_slicer
fig = horizon_slicer.make(df_paths)
fig.show()
```

### 12.65  Inset focus view
`viz.inset.make(fig, region)` embeds a zoomed-in inset within an existing figure
to highlight volatility spikes or crash periods. Pass a rectangle `(x0, x1, y0,
y1)` to define the zoom box.

### 12.66  Data-quality heatmap
`viz.data_quality.make(df_errors)` displays missing data or anomaly counts over
time, colouring cells by severity so data issues are obvious before plotting
performance. The DataFrame index should hold the observation date and columns
represent different fields.

### 12.67  Live update connector
`viz.live.connect(url, fig)` listens to a WebSocket feed and streams updates to
an existing figure. The Streamlit dashboard registers a callback to redraw
charts without refreshing the whole page. The helper is async and expects the
`websockets` package.

### 12.68  Bookmarkable figures
`viz.bookmark.save(fig)` returns a JSON blob capturing layout and traces. Load it
with `viz.bookmark.load(blob)` to recreate the figure exactly – handy for
emailing reproducible views or embedding JSON snapshots in the repo.

### 12.69  Notebook widget helpers
`viz.widgets.explore(df_summary)` wraps key chart functions in `ipywidgets`
controls so quants can tweak parameters interactively inside Jupyter. Sliders
let users apply simple transformations without rewriting the plotting code.

### 12.70  Interactive PDF export
`viz.pdf_export.save(fig, "chart.pdf")` embeds Plotly HTML in a self-contained
PDF so stakeholders can pan and zoom offline. The CLI `--pdf` flag delegates to
this helper when saving board-pack graphics.

### 12.71  Multi-page PDF report
`viz.pdf_report.save(figs, "report.pdf")` combines multiple figures into a
single document with one chart per page.  Use this when exporting an entire
scenario sweep so that board packs bundle neatly into a single file.

### 12.72  Real-time crossfiltering
`viz.crossfilter.make(figs, df)` links several charts together so selections in
one plot update the others. The helper wires up Plotly ``restyle`` callbacks and
returns a composite figure ready for the dashboard.

### 12.73  Hover-sync helper
`viz.hover_sync.apply(figs)` keeps hover labels aligned across multiple charts
showing different metrics.  Moving the cursor on one figure highlights the
corresponding month or scenario on every other figure in the list.

### 12.74  Surface slice explorer
`viz.surface_slice.make(df_grid, axis="AE_leverage")` lets PMs scroll through 3‑D
risk surfaces by fixing one parameter at a time.  A slider chooses the slice and
the resulting 2‑D heatmap updates live.

### 12.75  Dashboard layout templates
`viz.dashboard_templates.get(name)` returns a preconfigured Streamlit layout with
standard tabs and download buttons. Teams can extend these templates rather than
recreating common boilerplate for every new dashboard.

### 12.76  Monte-Carlo funnel
`viz.funnel.make(df_paths)` visualises how the distribution of cumulative
returns widens over time. The helper plots the median alongside configurable
quantile bands so the "funnel" of outcomes is immediately clear.

### 12.77  Surface animation
`viz.surface_animation.make(df_grid)` turns a parameter sweep into an animated
Plotly surface. Each frame represents a different value of the varying
parameter, letting PMs watch the risk/return landscape shift dynamically.

### 12.78  Grid panel
Arrange multiple figures on one canvas with `viz.grid_panel.make(figures,
cols=2)`. The helper wraps Plotly's `make_subplots` and applies the standard
theme so layout and fonts remain consistent.

### 12.79  Ranking table
`viz.rank_table.make(df_summary, metric="AnnReturn")` outputs a
`dash_table.DataTable` sorted by the chosen metric. Conditional formatting uses
the traffic-light thresholds from `config_thresholds.yaml` so winners and
laggards pop visually.

### 12.80  Scenario timeline player
`viz.scenario_play.make(df_paths)` adds play/pause buttons to step through
cumulative returns month by month. This manual control helps presenters walk an
audience through key events without scrubbing a video file.

### 12.81  Sparkline matrix
When analysing dozens of scenarios, `viz.spark_matrix.make(df_series)` renders a
grid of tiny sparkline charts. Each cell shows a mini cumulative-return curve so
patterns are immediately obvious.

### 12.82  Weighted stacked bars
`viz.weighted_stack.make(df)` draws stacked bars whose width encodes horizon
length. Use this view to compare capital deployment across overlapping time
windows.

### 12.83  Geographical exposure map
`viz.geo_exposure.make(df)` visualises exposures by region using Plotly's
`choropleth` or `scatter_geo`. Pass `Region` and `Exposure` columns to highlight
concentration risk across geographies.

### 12.84  Seasonality heatmap
`viz.seasonality_heatmap.make(df_paths)` arranges monthly returns by year and
month in a heatmap so seasonal patterns jump out.  The DataFrame index should be
a `(Year, Month)` MultiIndex or equivalent columns.

### 12.85  Beta–TE scatter
`viz.beta_te_scatter.make(df_summary)` plots beta exposure on the x‑axis against
tracking error on the y‑axis. Points inherit colours from
`viz.theme.CATEGORY_BY_AGENT` and scale by the `Capital` column.

### 12.86  Funding milestone timeline
Use `viz.milestone_timeline.make(events, fig)` to annotate key funding events on
top of the cumulative-return fan.  Each event is a `(month, label)` tuple and
the helper places a marker plus text box using the standard theme colours.

### 12.87  Outcome mosaic
`viz.mosaic.make(df_paths)` compresses final-return probabilities into a square
grid of coloured tiles.  This highlights skew and tail weight more intuitively
than a simple histogram.

### 12.88  Interactive metric selector
`viz.metric_selector.make(df_summary, metrics)` returns a small Streamlit widget
that lets users swap which metric (e.g. CVaR, Sharpe) appears on the
risk‑return scatter’s y‑axis without regenerating all figures.

### 12.89  Boxplot distribution
`viz.boxplot.make(df_paths)` draws a boxplot of monthly returns for each agent
so distribution asymmetry jumps out. Pass a `(n_sim, n_months)` array or a
mapping of agent → array. The helper colours each trace using
`viz.theme.CATEGORY_BY_AGENT`.

```python
from pa_core.viz import boxplot
fig = boxplot.make({"Base": df_base, "AE": df_ae})
fig.show()
```

### 12.90  Delta heatmap
Compare two parameter grids with `viz.delta_heatmap.make(df_a, df_b)`. Values in
`df_b` minus those in `df_a` are plotted as a diverging-colour heatmap so the
impact of tweaking leverage or capital shares is immediately clear.

### 12.91  Rolling quantile band
`viz.quantile_band.make(df_paths, quantiles=(0.1, 0.9), window=12)` overlays a
rolling high/low quantile ribbon on top of the median path. This highlights how
tail risk tightens or widens over time.

### 12.92  TE–Beta–Return 3‑D scatter
For multi-factor diagnostics use
`viz.triple_scatter.make(df_summary)` which plots tracking error, beta and
excess return on three axes. Marker size scales with the `Capital` column and
hover text lists CVaR.

### **13  CLI Additions** &nbsp;*(new subsection in cli.py docstring)*

// NEW  
```text
--png / --pdf / --pptx     Static exports (can be combined)
--html                    Save interactive HTML
--gif                     Animated export of monthly paths
--alt-text TEXT           Alt text for HTML/PPTX exports
--dashboard                Launch Streamlit dashboard after run

```
<!-----------------  🔧  CONFIG / METRIC UPGRADE  ---------------->
### :construction_worker: Task — ensure ShortfallProb is always produced

1 · **parameters template**
   * open `config/parameters_template.csv`
   * append a line (or update if it exists)
     ```
     risk_metrics,Return;Risk;ShortfallProb,Metrics to compute
     ```
   * propagate the same change to any YAML sample (`params_template.yml`).

2 · **CLI sanity-check** (`pa_core/config.py`)  
   * on load, assert `"ShortfallProb" in cfg.risk_metrics`;  
     if absent, raise `ConfigError("risk_metrics must include ShortfallProb")`.

3 · **Excel exporter** (`pa_core/reporting/excel.py`)  
   * before writing the *Summary* sheet:  
     ```python
     summary["ShortfallProb"] = summary.get("ShortfallProb", 0.0)
     ```
     so old output files never explode the viz.

4 · **Dashboard guard-rail** (`pa_core/viz/risk_return.py`)  
   * same one-liner:  
     ```python
     df = df.copy()
     df["ShortfallProb"] = df.get("ShortfallProb", 0.0)
     ```

5 · **Regression test** (`tests/test_outputs.py`)
   ```python
   import pandas as pd, pathlib
   def test_shortfall_present():
       fn = pathlib.Path("Outputs.xlsx")
       assert fn.exists(), "Outputs.xlsx missing"
       cols = pd.read_excel(fn, sheet_name="Summary").columns
       assert "ShortfallProb" in cols

6 · **Ongoing maintenance**
   * Keep refining both `parameters_template.csv` and `params_template.yml`
     until they contain every value needed to run the entire simulation suite
     without manual edits.
