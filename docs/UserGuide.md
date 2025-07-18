# Portable Alpha-Extension Model User Guide

This program simulates a portable‑alpha plus active‑extension strategy. Each run
distributes capital across internal, external portable‑alpha and active‑extension
sleeves and draws joint return paths. The command line prints a summary and
writes an Excel workbook with an embedded risk‑return chart along with optional
additional figures. Use ``python -m pa_core.cli`` to access all command‑line
features—including dashboard launch and static exports. The parameter templates
in `config/` already include the mandatory `ShortfallProb` risk metric so the
CLI will fail fast if you remove it. All tutorials assume you invoke the program
All parameter files include an `analysis_mode` field selecting `returns`, `capital`, `alpha_shares` or `vol_mult`. The CLI validates this value.
via ``python -m pa_core.cli``.

The model is designed to help you explore three core ideas:

* **Risk/return trade‑offs** – how annualised return compares with volatility.
* **Funding shortfall probability** – the chance the portfolio falls below a required level.
* **Tracking error** – deviation from the benchmark.

The introductory tutorials demonstrate how to implement a run, interpret these metrics and visualise them so you can test each idea in a repeatable workflow.

### Quick start

1. **Implement a scenario** – run the model with a CSV or YAML config.
2. **Interpret the metrics** – review the summary table and check `ShortfallProb` and `TrackingErr`.
3. **Visualise the results** – launch the dashboard or use `scripts/visualise.py`.

Example quick run:

```bash
# Install dependencies once
pip install -r requirements.txt

# Run a 500-path, 15-year simulation of all agents
python main.py \
  --agent ExternalPA ActiveExt InternalPA InternalBeta Base \
  --n_sims 500 --n_months $((15*12)) \
  --save_xlsx Outputs.xlsx \
  --seed 42
```

### Key concepts

* **Risk/return trade‑off** – compare annualised return and volatility across sleeves.
* **Funding shortfall risk** – monitor the required `ShortfallProb` metric (include it under `risk_metrics` or configuration fails).
* **Tracking error** – check how far each sleeve deviates from the benchmark.
* **Visualisation** – explore results via the dashboard or scripts.
* **Scenario testing** – alter capital weights or alpha assumptions to see the impact on all metrics.

## 1. Overview

The model allocates capital across three sleeves—internal, external portable‑alpha
and active‑extension—and simulates monthly returns for each. Key outputs include
annualised return, volatility, Value at Risk, tracking error and the required
**ShortfallProb** metric. The tutorials below walk through running a simulation,
interpreting these metrics and visualising them so you can test the model’s main
ideas in practice. Each tutorial highlights how to implement a run, review the
headline metrics and visualise the results so you can evaluate risk/return,
shortfall probability and tracking error in a repeatable workflow.

## 2. Getting Started

1. Run `./setup.sh` once to create a virtual environment and install all dependencies.
2. The script installs **Streamlit** for the dashboard and **Kaleido** for static exports so no extra packages are required.
3. Copy `config/parameters_template.csv` or `config/params_template.yml` and edit the values to suit your scenario. Launch the CLI with `--params` or `--config` and supply your index returns via `--index`.
4. **Review defaults** – core correlations and volatilities are locked in `pa_core/config.py`. Override them in your parameter file only when testing different assumptions.
5. Set the **Analysis mode** in your parameter file to `returns`, `capital`, `alpha_shares` or `vol_mult`. The templates default to `returns`.
6. The index CSV must contain a `Date` column and either `Monthly_TR` or `Return` for monthly total returns.
7. Make sure your parameter file includes `ShortfallProb` under `risk_metrics`; removing it triggers a validation error.
   Older output files that predate this requirement will still load—both the Excel
   exporter and dashboard insert a `ShortfallProb` column with `0.0` so legacy
   results remain compatible.
8. Add `--seed` for reproducible draws or `--backend cupy` if a GPU is available.
9. When a seed is supplied the program uses `spawn_agent_rngs` to create
   deterministic random-number generators per sleeve so results are fully
   repeatable.
10. **Financing spikes** are controlled via `internal_spike_prob`, `ext_pa_spike_prob` and `act_ext_spike_prob`. Set them to `0.0` for a simplified first run.
11. Run `python -m pa_core.cli --help` at any time to view all command-line options.
12. Include `--dashboard` to open an interactive Streamlit view after the run completes. The dashboard now offers an **Auto‑refresh** checkbox so you can reload results periodically while long simulations run.
13. Install Chrome or Chromium if you plan to use `--png`, `--pdf` or `--pptx`; these exports rely on the browser together with Kaleido.

```bash
python -m pa_core.cli --params parameters.csv --index sp500tr_fred_divyield.csv
```
The run prints a console summary and writes an Excel workbook (`Outputs.xlsx` by default). If you include the `--pivot` flag the raw return paths are also saved in an
`AllReturns` sheet. The existing sheet order is retained for compatibility. Convert the extra sheet to an `Outputs.parquet` file and keep it alongside the Excel workbook whenever you want the dashboard to display path‑based charts.
## 3. Introductory Tutorial Series

The following tutorials provide a hands‑on introduction to the model.
Work through them in order to learn how to implement a run, interpret
the key metrics and visualise the results.  This quick start teaches you
how to evaluate the model’s three main purposes—**risk/return trade‑offs**,
**funding shortfall probability** and **tracking error**—in a repeatable
workflow.

### Tutorial roadmap

1. **Introductory Tutorial 1 – Master the Program (5 Parts)**
   - **Part 1**: Basic Program Operation - single scenario fundamentals
   - **Part 2**: Capital Mode - allocation percentage sweeps
   - **Part 3**: Returns Mode - return/volatility sensitivity analysis
   - **Part 4**: Alpha Shares Mode - alpha/beta split optimization
   - **Part 5**: Vol Mult Mode - volatility stress testing
2. **Introductory Tutorial 2 – Interpret the Metrics** – review `AnnReturn`, `AnnVol`, `ShortfallProb` and `TrackingErr` in the console and workbook.
3. **Introductory Tutorial 3 – Visualise the Results** – launch the dashboard or notebook to explore the risk‑return scatter, funding fan and return distribution.
   These first three tutorials form a quick‑start sequence for testing the core ideas.
4. **Export Charts** – save PNG, PDF, PPTX, HTML or GIF figures directly from the CLI.
5. **Generate Custom Visualisations** – use `scripts/visualise.py` on saved outputs.
6. **Implement a New Agent** – subclass `BaseAgent` and register it.
7. **Customise Visual Style** – adjust YAML theme files and reload.
8. **Stress-Test Your Assumptions** – run multiple scenarios and compare metrics.
9. **Save Everything with Export Bundles** – archive figures via `viz.export_bundle`.
10. **Explore the Chart Gallery** – open `viz_gallery.ipynb` for a hands-on tour of every plotting helper.

### Parameter Sweep Engine

Use `--mode` to run automated sweeps across common parameters.  The CLI writes a
spreadsheet summarising each combination and embeds a risk‑return chart using
`viz.risk_return.make()` so results align with the single‑run workflow.
All sweep results feed into the same visualisation functions as single runs—
helpers like `risk_return`, `surface` and `fan` therefore work across every
analysis mode without modification.  Parameter ranges are supplied via a YAML or
CSV file so the CLI can iterate through each scenario automatically.

#### Capital mode
- **Purpose:** Test different allocation mixes between the external PA and
  active‑extension sleeves.
- **Key parameters:** `external_pa_capital`, `active_ext_capital`,
  `internal_pa_capital` (must sum to the total fund size).
- **CLI example:**
  ```bash
  python -m pa_core.cli --mode capital --config my_params.yml
  ```
  Example parameter file snippet:
  ```yaml
  analysis_mode: capital
  external_pa_capital: [80, 100]
  active_ext_capital: [20, 40]
  internal_pa_capital: [100]
  ```

#### Returns mode
- **Purpose:** Explore how varying return and volatility assumptions affects
  portfolio metrics.
- **Key parameters:** `exp_return_*` and `vol_*` values in the config file.
- **CLI example:**
  ```bash
  python -m pa_core.cli --mode returns --config my_params.yml
  ```
  Example snippet:
  ```yaml
  analysis_mode: returns
  exp_return_H: [0.04, 0.05]
  vol_H: [0.01]
  ```

#### Alpha shares mode
- **Purpose:** Sweep the alpha/beta splits for external PA and active extension
  managers.
- **Key parameters:** `external_pa_alpha_fraction`, `active_share`.
- **CLI example:**
  ```bash
  python -m pa_core.cli --mode alpha_shares --config my_params.yml
  ```
  Example snippet:
  ```yaml
  analysis_mode: alpha_shares
  external_pa_alpha_fraction: [0.4, 0.6]
  active_share: [0.5, 0.7]
  ```

#### Vol mult mode
- **Purpose:** Stress test the strategy under higher volatility regimes by
  applying multipliers to all volatilities.
- **Key parameters:** `vol_mult` values such as `1.5, 2.0, 2.5`.
- **CLI example:**
  ```bash
  python -m pa_core.cli --mode vol_mult --config my_params.yml
  ```
  Example snippet:
  ```yaml
  analysis_mode: vol_mult
  vol_mult: [1.5, 2.0, 2.5]
  ```

Each sweep mode produces one Excel sheet per parameter combination and an
overall summary table with a risk‑return scatter generated by
`viz.risk_return.make()`.  Start with the CSV templates in `config/` for
ready‑to‑run examples.  The summary sheet lists every scenario so you can quickly
rank them by `AnnReturn` or `ShortfallProb`.

### Interpreting sweep results

After running a sweep the output workbook (e.g. `Sweep.xlsx`) contains one
sheet per parameter combination alongside a `summary` sheet. This sheet
aggregates the headline metrics and embeds a risk‑return scatter so you can
compare scenarios quickly. Sort the table by `AnnReturn` or
`ShortfallProb` to locate the most promising cases.  The same summary DataFrame
works with `viz.surface.make()` or `viz.grid_heatmap.make()` to visualise the
risk/return landscape.
Open the individual sheets named after each parameter set to inspect detailed
metrics.  The sheet names follow the pattern `param=value` so they remain easy to
match against the summary table.

Example command:

```bash
python -m pa_core.cli --mode capital --config sweep_capital.yml --output Sweep.xlsx
```

Introductory Tutorials 1‑3 cover the main workflow of implementing a scenario, interpreting the output metrics and visualising risk/return, funding shortfall and tracking error. Later tutorials introduce exports, customisation and stress-testing.

### Introductory Tutorial 1 – Master the Program (5 Parts)

This comprehensive tutorial introduces you to both basic operation and the powerful parameter sweep capabilities. Work through all five parts in order to build complete understanding.

#### Part 1: Basic Program Operation

**Objective**: Run your first simulation and understand the fundamental output structure.

As a new user, start with the simplest possible command to establish baseline understanding:

1. **Copy the basic template**:
   ```bash
   cp config/params_template.yml my_first_scenario.yml
   ```

2. **Run your first simulation** (single scenario, no sweep):
   ```bash
   python -m pa_core.cli \
     --config my_first_scenario.yml \
     --index sp500tr_fred_divyield.csv \
     --output MyFirstResults.xlsx
   ```

3. **Understand the console output**: You'll see a Rich table showing:
   - `AnnReturn`: Annualized return percentage for each sleeve
   - `AnnVol`: Annualized volatility (risk measure)
   - `VaR`: Value at Risk at 5% level
   - `BreachProb`: Probability of funding shortfall
   - `TE`: Tracking Error relative to benchmark

4. **Examine the Excel file**: Open `MyFirstResults.xlsx` to see:
   - **Summary Sheet**: Key metrics for all sleeves
   - **Inputs Sheet**: Confirms your configuration parameters
   - **Risk-Return Chart**: Visual representation embedded in Excel

**Success Check**: You should see results for 3-4 sleeves (Internal PA, External PA, Active Extension, etc.) with realistic financial metrics.

#### Part 2: Capital Mode - Allocation Sweeps

**Objective**: Understand how parameter sweeps work by varying capital allocations.

The `--mode=capital` parameter runs multiple scenarios automatically, varying external and active extension capital allocations:

1. **Examine the capital template**:
   ```bash
   head config/capital_mode_template.csv
   ```
   You'll see columns for different capital allocation scenarios.

2. **Run a capital allocation sweep**:
   ```bash
   python -m pa_core.cli \
     --config config/capital_mode_template.csv \
     --index sp500tr_fred_divyield.csv \
     --mode capital \
     --output CapitalSweep.xlsx
   ```

3. **Compare the results**: Notice that `CapitalSweep.xlsx` now contains:
   - Multiple sheets (one per allocation scenario)
   - Summary sheet with all combinations
   - Risk-return chart showing the efficient frontier

**Key Insight**: Capital mode helps you find optimal allocation percentages by testing multiple combinations automatically.

#### Part 3: Returns Mode - Sensitivity Analysis

**Objective**: Explore how different return and volatility assumptions affect outcomes.

Use `--mode=returns` to test various return/volatility scenarios:

1. **Examine the returns template**:
   ```bash
   head config/returns_mode_template.csv
   ```
   This template varies expected returns and volatilities for different agents.

2. **Run returns sensitivity analysis**:
   ```bash
   python -m pa_core.cli \
     --config config/returns_mode_template.csv \
     --index sp500tr_fred_divyield.csv \
     --mode returns \
     --output ReturnsSweep.xlsx
   ```

3. **Interpret the results**: The sweep shows how sensitive your strategy is to return assumptions. Higher expected returns generally increase both returns and risks.

**Key Insight**: Returns mode helps stress-test your assumptions about future market performance.

#### Part 4: Alpha Shares Mode - Optimization

**Objective**: Understand alpha vs. beta share allocation and its impact on tracking error.

Use `--mode=alpha_shares` to optimize the split between alpha-generating and beta-matching components:

1. **Examine the alpha shares template**:
   ```bash
   head config/alpha_shares_mode_template.csv
   ```
   This varies the percentage allocated to alpha generation vs. beta matching.

2. **Run alpha/beta optimization**:
   ```bash
   python -m pa_core.cli \
     --config config/alpha_shares_mode_template.csv \
     --index sp500tr_fred_divyield.csv \
     --mode alpha_shares \
     --output AlphaSweep.xlsx
   ```

3. **Analyze tracking error trade-offs**: Higher alpha allocation may increase returns but also tracking error.

**Key Insight**: Alpha shares mode helps balance return enhancement with tracking error constraints.

#### Part 5: Vol Mult Mode - Stress Testing

**Objective**: Perform comprehensive stress testing by scaling volatilities.

Use `--mode=vol_mult` to test how your strategy performs under different volatility regimes:

1. **Examine the volatility multiplier template**:
   ```bash
   head config/vol_mult_mode_template.csv
   ```
   This scales all volatilities by different multipliers (e.g., 0.5x, 1.0x, 1.5x, 2.0x).

2. **Run volatility stress test**:
   ```bash
   python -m pa_core.cli \
     --config config/vol_mult_mode_template.csv \
     --index sp500tr_fred_divyield.csv \
     --mode vol_mult \
     --output VolStressTest.xlsx
   ```

3. **Evaluate resilience**: See how your strategy performs in low, normal, and high volatility environments.

**Key Insight**: Vol mult mode reveals how robust your strategy is to changing market volatility.

#### Tutorial 1 Summary

You've now mastered:
- ✅ Basic single-scenario operation (Part 1)
- ✅ Capital allocation optimization (Part 2)
- ✅ Return assumption sensitivity (Part 3)
- ✅ Alpha/beta split optimization (Part 4)
- ✅ Volatility stress testing (Part 5)

**Next Steps**: Proceed to Tutorial 2 to learn detailed metric interpretation, or Tutorial 3 to explore the interactive dashboard. All visualization features work with results from any of these five approaches.

**Troubleshooting**:
- If commands fail, ensure you're in the correct directory and virtual environment is activated
- Check that CSV templates exist in the `config/` folder
- Verify `sp500tr_fred_divyield.csv` is present in the root directory
- Use `python -m pa_core.cli --help` to see all available options

### Introductory Tutorial 2 – Interpret the Metrics (Risk/Return, Shortfall and Tracking Error)

This tutorial explains how to read the results produced in Tutorial 1 **(any of the 5 parts)**. Whether you ran a single scenario (Part 1) or parameter sweeps (Parts 2–5), the core metrics remain the same. After running the model you will see a Rich table of headline metrics and an Excel workbook of detailed results capturing risk/return profile, funding shortfall probability and tracking error for each sleeve.
Work through the following steps to interpret the results:

1. **Open `Outputs.xlsx`** – check the `Inputs` sheet to confirm your scenario
   parameters and locate the `Summary` sheet.
2. **Review the headline metrics** – `AnnReturn`, `AnnVol`, `VaR`, `BreachProb`, `ShortfallProb` and `TE` appear for each sleeve. The workbook also includes the mandatory **ShortfallProb** column even if it was not requested in your configuration.
   The sample configuration intentionally uses high leverage so you may notice
   **TE** values near **8–10%**, well above the default **3%** budget. Treat this
   as a red flag and use it to practise threshold analysis.
3. **Compare to thresholds** – verify `ShortfallProb` against the limits defined
   in `config_thresholds.yaml` and examine `TE` to ensure each sleeve stays
   within your tracking‑error budget.

4. **Analyse parameter sweep results** – when running a sweep the `Summary`
   sheet lists every scenario. Sort the table by `TE` and
   `ShortfallProb` to identify combinations that meet the default
   **3% tracking‑error cap**. Add a pivot table or filter by `TE < 0.03` and
   `ShortfallProb` to highlight combinations where **all** sleeves stay below
   the threshold. With 50–200 scenarios this method quickly reveals which
   parameters cause breaches across multiple sheets.

5. **Interpret the risk levels** – values below **1% TE** generally fall
   within a conservative comfort zone, **1–3%** indicates moderate risk
   and **above 3%** breaches the default budget. Use conditional
   formatting to colour scenarios amber or red when they cross these
   lines.

6. **Take action when limits are breached** – scenarios with `TE` above the
   budget or high `ShortfallProb` usually need lower leverage or a
   different capital allocation. Iterate on the parameter sweep by
   adjusting the input template and rerunning until the majority of cases
   comply with your risk limits.

`ShortfallProb` is a mandatory metric. If you omit it from `risk_metrics` the
CLI raises a validation error. The dashboard uses the same threshold file so
colours remain consistent.

### Introductory Tutorial 3 – Visualise the Results (Dashboard and Scripts)

This tutorial shows how to visualise the metrics produced in Tutorial 1 (all 5 parts) and Tutorial 2. The dashboard works with results from any mode—single scenarios, capital sweeps, returns analysis, alpha optimization or volatility stress tests. After generating an output file you can start an interactive dashboard to explore the portfolio behaviour visually. Follow these steps:

1. **Launch the dashboard** – either add `--dashboard` to the CLI call or run
   `streamlit run dashboard/app.py` manually. If you installed the package
   manually make sure `streamlit` is available:
   ```bash
   pip install streamlit
   ```
2. **Load your results** – enter the path to `Outputs.xlsx` in the sidebar. If a
   matching `Outputs.parquet` file exists the dashboard enables additional
   charts. Parameter sweep files (typically 38–183 KB) automatically expose a
   **Scenario** selector so you can browse dozens of combinations.
3. **Add path data for advanced charts** – include the `--pivot` flag when
   running the CLI so an `AllReturns` sheet is saved. Convert that sheet to
   `Outputs.parquet` using pandas to unlock the funding fan and distribution
   views.
4. **Explore the tabs** – the headline view shows a risk‑return scatter while
   other tabs display cumulative funding (`Funding fan`) and final return
   distributions (`Path dist`). When a sweep file is loaded use the **Scenario**
  dropdown to compare up to 200 combinations. Threshold lines from
  `config_thresholds.yaml` highlight compliant cases. Two download buttons let
  you save the headline PNG chart and the Excel file directly from the browser.
  Tick **Auto‑refresh** to reload the data periodically while a long simulation
   runs. **PNG downloads require a local Chrome/Chromium installation and the**
   **`kaleido`** **package**, otherwise the export button will fail silently.

### Sidebar Controls

- **Results file** – text box to locate the Excel output.
- **Months** – slider to limit the number of months shown.
- **Agents** – choose which sleeves to display.
- **Risk‑free rate** – for any excess return calculations.
- **Auto‑refresh** – polls the file every few seconds so the dashboard updates while simulations run.
- **Scenario** – appears when a parameter sweep file is loaded; pick a combination to display in the charts.

Two download buttons allow you to save the headline PNG chart and the Excel file.

### Introductory Tutorial 4 – Exporting Charts

The CLI can create static images or PPTX packs as part of a run. Combine the following flags as needed:

```text
--png  --pdf  --pptx  --html  --gif  --alt-text "Description"
```

`--html` saves an interactive Plotly page, while `--gif` exports an animation of monthly paths.  The optional `--alt-text` flag attaches descriptive text to HTML and PPTX exports so charts remain accessible.  You can also run `scripts/visualise.py` after a simulation to generate additional charts from the saved output files.

> **Dependency Note**
> PNG/PDF/PPTX exports require a local Chrome or Chromium installation in addition to the
> `kaleido` Python package. Install Chrome with:
> ```bash
> sudo apt-get install -y chromium-browser
> ```

### Tutorial 5 – Generate Custom Visualisations

Use `scripts/visualise.py` to build plots outside the dashboard. The script
reads the Excel output along with an optional `.parquet` file of raw paths and
can export any Plotly figure. Some plots such as `fan` and `path_dist` require
the Parquet file, so convert the `AllReturns` sheet if you plan to use them.
Pass one of the following names to `--plot`:
`risk_return`, `risk_return_bubble`, `fan`, `path_dist`, `corr_heatmap`,
`sharpe_ladder`, `rolling_panel`, `rolling_var`, `breach_calendar`, `overlay`,
`overlay_weighted`, `category_pie`, `gauge`, `waterfall`, `surface`,
`grid_heatmap`, `scenario_slider`, `scenario_viewer`, `surface_animation`,
`surface_slice`, `crossfilter`, `hover_sync`, `grid_panel`, `scenario_play`,
`spark_matrix`, `weighted_stack`, `geo_exposure`, `seasonality_heatmap`,
`beta_te_scatter`, `milestone_timeline`, `mosaic`, `metric_selector`,
`boxplot`, `delta_heatmap`, `quantile_band`, `triple_scatter`,
`radar`, `scatter_matrix`, `parallel_coords`, `data_table`,
`capital_treemap`, `corr_network`, `beta_heatmap`, `factor_bar`,
`multi_fan`, `beta_scatter`, `factor_matrix`, `te_cvar_scatter`,
`quantile_fan`, `sunburst`, `horizon_slicer`, `inset`, `data_quality`,
`live`, `bookmark`, `widgets`, `pdf_export`, `pdf_report`, `funnel`,
`rank_table`, `exposure_timeline`, `rolling_corr_heatmap`,
`moments_panel`, `factor_timeline`.

Common plot types and when to use them:

- `risk_return` – scatter of annualised return vs volatility with TE/ER guides.
- `fan` – cumulative-return fan with median and confidence bands.
- `path_dist` – histogram or CDF of final returns to check skew.
- `corr_heatmap` – monthly-return correlation matrix.
- `sharpe_ladder` – bar ranking agents by Sharpe ratio.
- `rolling_panel` – drawdown, TE and Sharpe metrics over time.
- `surface` – 3‑D risk/return surface from parameter sweeps.
- `category_pie` – donut summarising capital allocation by agent.
- `overlay` – line chart comparing cumulative paths across agents.
- `waterfall` – bar series illustrating capital contributions step by step.
- `gauge` – single-value dial highlighting risk or return versus a target.
- `moments_panel` – dashboard view of rolling mean, volatility and skew.
- `rolling_corr_heatmap` – time-series heatmap showing correlation drift.
Combine with `--png`, `--pdf`, `--pptx`, `--html`, `--gif` and an optional
`--alt-text` description to save images:

```bash
python scripts/visualise.py \
  --plot overlay \
  --xlsx Outputs.xlsx \
  --png --alt-text "Overlay of median paths"
```

To animate precomputed frames, try the `scenario_slider` plot:

```bash
python scripts/visualise.py \
  --plot scenario_slider \
  --xlsx Outputs.xlsx \
  --gif --alt-text "Scenario slider"
```
GIF exports rely on Chrome and Kaleido. If these are missing the script logs a
warning and no file is created. Check the console output for these warnings if
no GIF appears.

If your Excel file includes an `AllReturns` sheet, convert it to Parquet first:

```python
import pandas as pd
df = pd.read_excel("Outputs.xlsx", sheet_name="AllReturns")
df.to_parquet("Outputs.parquet")
```

Place both files in the same folder and rerun the script to access path based
charts such as the funding fan or return histogram.

### Tutorial 6 – Implement a New Agent

1. Create a new class under `pa_core/agents/` that subclasses `BaseAgent` and implement `monthly_returns` to return an `(n_sim, n_months)` array.
2. Register the class in `_AGENT_MAP` inside `pa_core/agents/registry.py`.
3. **Extend the configuration** – add a field to `ModelConfig` so capital can be allocated via YAML/CSV:
   ```python
   class ModelConfig(BaseModel):
       my_agent_capital: float = 0.0
   ```
   Update `build_from_config` in `pa_core/agents/registry.py` to create an `AgentParams` entry when `my_agent_capital` is positive.
4. Include `my_agent_capital` in your configuration template and set the amount you want to allocate.
5. Run the CLI again and the sleeve will automatically appear in the outputs.

```python
# pa_core/agents/my_agent.py
from pa_core.agents.base import BaseAgent

class MyAgent(BaseAgent):
    def monthly_returns(self, r_beta, alpha_stream, financing):
        # compute returns for each simulation and month
        return r_beta + alpha_stream - financing
```

### Tutorial 7 – Customise Visual Style

Colours, fonts and traffic-light thresholds load from `config_theme.yaml` and `config_thresholds.yaml`. Edit these files before running the CLI or dashboard to adjust palettes or risk limits. After editing, reload the dashboard or call `pa_core.viz.theme.reload_theme()` from Python.
The theme file now accepts `paper_bgcolor` and `plot_bgcolor` keys so you can
match the dashboard background to corporate colours.

```yaml
# config_thresholds.yaml
shortfall_green: 0.05
shortfall_amber: 0.10
```

### Example workflow

To apply a custom theme when analysing a parameter sweep, load your YAML file
before generating figures:

```python
from pa_core.viz import theme, risk_return
import pandas as pd

theme.load_theme("my_theme.yaml")
df_summary = pd.read_excel("Sweep.xlsx", sheet_name="Summary")
fig = risk_return.make(df_summary)
fig.write_image("plots/sweep.png")
```
Reload different theme files inside a loop to style each scenario in a
parameter sweep automatically.

## Dashboard Views

The dashboard contains four tabs, each aimed at a different angle on portfolio behaviour.

### Headline

Displays a **risk‑return scatter** where each agent is a coloured marker. The "sweet‑spot" rectangle highlights the desired tracking‑error and excess‑return range. Use this view to compare risk and expected return across investment options.

### Funding fan

Shows the distribution of cumulative returns as a ribbon around the median path. The widening band illustrates potential funding shortfall or surplus over time. This helps gauge drawdown risk.

### Path dist

Plots a histogram of final returns with an optional CDF overlay. Switch views to estimate the probability of breaching a given return level.

### Diagnostics

Lists the raw summary table for reference and allows quick export.

These visual tools complement the Excel output by making it easy to spot how reallocating capital or adjusting assumptions shifts the risk/return profile of the three sleeves.

### Tutorial 8 – Stress-Test Your Assumptions

After completing the tutorials you can stress‑test your assumptions by **automating parameter sweeps**. Supply one of the sweep templates and choose a mode such as `returns` or `capital` to run dozens of scenarios in a single command. Always set `--output` so the results are saved to a unique workbook instead of overwriting `Outputs.xlsx`:

```bash
python -m pa_core.cli \
  --mode returns \
  --config config/returns_mode_template.csv \
  --output StressTest.xlsx
```

Open the `Summary` sheet to filter by `TE` and `ShortfallProb`, or load the file in the dashboard where the **Scenario** selector lets you compare paths and metrics side by side.

### Tutorial 9 – Save Everything with Export Bundles

Use the `viz.export_bundle.save` helper to output PNG, HTML and JSON files for multiple figures at once. Pass a list of figures and a file stem:

```python
from pa_core.viz import export_bundle, risk_return, fan
figs = [risk_return.make(df_summary), fan.make(df_paths)]
export_bundle.save(figs, "plots/summary")
```

The helper writes the files under `plots/` so you can archive an entire run with one call. Combine this with the CLI export flags when you need a full set of images.

When working with parameter sweeps, loop over each scenario to save figures with meaningful names:

```python
for label, df in sweep_dfs.items():
    figs = [risk_return.make(df), fan.make(paths[label])]
    export_bundle.save(figs, f"plots/{label}")
```

This produces a folder of charts for every scenario in the sweep so you can compare results side by side.

### Tutorial 10 – Explore the Chart Gallery

A Jupyter notebook `viz_gallery.ipynb` at the project root demonstrates every chart function with sample data. Launch it after installing the package in editable mode:

```bash
pip install -e .
jupyter notebook viz_gallery.ipynb
```

Adjust the parameters in the notebook to see how colours and thresholds react. The gallery is a quick way to experiment with new scenarios and visual styles without extra code.
Recent additions include advanced plots such as `overlay`, `category_pie`,
`gauge`, `waterfall`, `moments_panel`, `scenario_slider`, `scenario_viewer`,
`scenario_play`, `spark_matrix`, `weighted_stack`, `geo_exposure`,
`seasonality_heatmap`, `beta_te_scatter`, `milestone_timeline`, `mosaic`,
`metric_selector`, `boxplot`, `delta_heatmap`, `quantile_band` and
`triple_scatter`. Browse the notebook to see example usage of each helper along
with interactive widgets like `crossfilter` and `hover_sync`.

### Tutorial 11 – Run Parameter Sweeps

The model now supports automated sweeps across key parameters. Supply a CSV or YAML file with lists of values and set `--mode` to iterate through each combination. The CLI writes a workbook summarising every run and embeds a risk‑return scatter so you can compare scenarios quickly.

Example command:

```bash
python -m pa_core.cli --mode returns --config sweep_returns.yml --output Sweep.xlsx
```

Each sheet in `Sweep.xlsx` corresponds to one parameter set while the `summary` sheet aggregates the metrics. Use helpers like `viz.surface.make()` or `viz.grid_heatmap.make()` on the summary table to visualise the landscape.

### Tutorial 12 – Export a PowerPoint Report

Need presentation-ready slides? Include the `--pptx` flag when running the CLI and a `board_pack.pptx` file will be created with the headline charts.

```bash
python -m pa_core.cli --config my_params.yml --index sp500tr_fred_divyield.csv --pptx
```

To customise which figures appear, call the helper in Python:

```python
from pa_core.viz import pptx_export, risk_return, category_pie
figs = [risk_return.make(df_summary), category_pie.make(capital_map)]
pptx_export.save(figs, "my_pack.pptx")
```

Use `--alt-text` with the CLI if you want descriptive captions embedded for accessibility.

### Tutorial 13 – Weighted Risk‑Return Bubble

When sleeve capital varies it can help to scale each point by investment weight.
`viz.risk_return_bubble.make` behaves like `risk_return.make` but expects a
`Capital` column and draws each marker sized by that value. Use this view to
highlight which agents dominate the portfolio.

```python
from pa_core.viz import risk_return_bubble
fig = risk_return_bubble.make(df_summary)
fig.show()
```

### Tutorial 14 – Rolling VaR Timeline

Visualise how tail risk evolves over time with
`viz.rolling_var.make(df_paths, window=12)`. The helper computes the
rolling Value at Risk (or CVaR) over the chosen horizon and plots it as a
line chart.

```python
from pa_core.viz import rolling_var
fig = rolling_var.make(df_paths, window=12)
fig.show()
```
