# Portable Alpha-Extension Model User Guide

This program simulates a portable‑alpha plus active‑extension strategy. Each run distributes capital across internal, external portable‑alpha and active‑extension sleeves and draws joint return paths. The command line prints a summary and writes an Excel workbook along with optional charts. Use ``python -m pa_core.cli`` to access all command-line features including dashboard launch and static exports. The model is primarily used to test risk/return trade‑offs, funding shortfall probability and tracking error across multiple scenarios. The parameter templates in `config/` already include the mandatory `ShortfallProb` risk metric so the CLI will fail fast if you remove it. All tutorials assume you invoke the program via ``python -m pa_core.cli``.

### Key concepts

* **Risk/return trade‑off** – compare annualised return and volatility across sleeves.
* **Funding shortfall risk** – monitor the required `ShortfallProb` metric (this column is added automatically if missing).
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
2. Copy `config/parameters_template.csv` or `config/params_template.yml` and edit the values to suit your scenario. Launch the CLI with `--params` or `--config` and supply your index returns via `--index`.
3. The index CSV must contain a `Date` column and either `Monthly_TR` or `Return` for monthly total returns.
4. Make sure your parameter file includes `ShortfallProb` under `risk_metrics`; removing it triggers a validation error.
5. Add `--seed` for reproducible draws or `--backend cupy` if a GPU is available.
6. Run `python -m pa_core.cli --help` at any time to view all command-line options.

```bash
python -m pa_core.cli --params parameters.csv --index sp500tr_fred_divyield.csv
```
The run prints a console summary and writes an Excel workbook (`Outputs.xlsx` by default). If you include the `--pivot` flag the raw return paths are also saved in an
`AllReturns` sheet. Convert this sheet to an `Outputs.parquet` file and keep it alongside the Excel workbook whenever you want the dashboard to display path‑based charts.
## 3. Introductory Tutorials

The following tutorials show how to implement a run, interpret the core metrics that measure risk/return, funding shortfall and tracking error, and visualise the results. Work through them in order the first time you use the program.

### Tutorial roadmap

1. **Implement the Model** – run a simulation from a parameter file and confirm the generated Excel workbook.
2. **Interpret the Metrics** – review the console table and workbook to understand risk/return, shortfall probability and tracking error.
3. **Visualise the Results** – launch the dashboard or notebook to explore interactive charts.
   These first three tutorials form a quick-start sequence for testing the core ideas.
4. **Export Charts** – save PNG, PDF, PPTX, HTML or GIF figures directly from the CLI.
5. **Generate Custom Visualisations** – use `scripts/visualise.py` on saved outputs.
6. **Implement a New Agent** – subclass `BaseAgent` and register it.
7. **Customise Visual Style** – adjust YAML theme files and reload.
8. **Stress-Test Your Assumptions** – run multiple scenarios and compare metrics.
9. **Save Everything with Export Bundles** – archive figures via `viz.export_bundle`.
10. **Explore the Chart Gallery** – open `viz_gallery.ipynb` for a hands-on tour of every plotting helper.

Tutorials 1-3 cover the main workflow of implementing a scenario, interpreting the output metrics and visualising risk/return, funding shortfall and tracking error. Later tutorials introduce exports, customisation and stress-testing.

### Tutorial 1 – Implement the Model

Start by editing one of the templates in `config/` or create your own CSV of parameters. Run the CLI to generate an Excel workbook for a single scenario. Use `--output` to change the filename and `--pivot` to append raw returns. This first tutorial walks through implementing the model and verifying the output:

1. Create or edit a parameter file.
2. Run the CLI with the parameter file and an index return CSV.
3. Confirm that `Outputs.xlsx` appears and includes a `Summary` sheet.
4. Check that the sheet lists `ShortfallProb` alongside `AnnReturn`, `AnnVol`, `VaR`, `TE` and `BreachProb`.

```bash
python -m pa_core.cli \
  --config my_params.yml \
  --index sp500tr_fred_divyield.csv \
  --output Results.xlsx \
  --pivot
```

Set `--seed` for reproducible draws or `--backend cupy` if a GPU is available. This first run produces a console table showing `AnnReturn`, `AnnVol`, `VaR`, `TE` and `BreachProb` for each sleeve and writes the same data to `Outputs.xlsx`.

### Tutorial 2 – Interpret the Metrics (Risk/Return, Shortfall and Tracking Error)

After running the model you will see a Rich table of headline metrics and an Excel workbook of detailed results. The workbook summarises **AnnReturn**, **AnnVol**, **VaR**, **TE**, **BreachProb** and a **ShortfallProb** column derived from that breach probability. Review the `Inputs` sheet to confirm parameters and the `Summary` sheet to compare sleeves. Use these metrics to interpret the simulation:

1. Compare **AnnReturn** and **AnnVol** for each sleeve to gauge the risk/return balance.
2. Check **ShortfallProb** against the thresholds defined in `config_thresholds.yaml`.
3. Use **TrackingErr** to ensure each sleeve stays within permitted deviation from the benchmark.

`ShortfallProb` is required by the program and will be added automatically if
your configuration omits it. The dashboard uses the same threshold file so
colours remain consistent.

### Tutorial 3 – Visualise the Results (Dashboard and Scripts)

After producing an output file you can start an interactive dashboard to explore the portfolio behaviour visually. The dashboard helps interpret risk/return trade‑offs, funding shortfall and tracking error across sleeves.

### Option 1 – from the CLI

Add `--dashboard` to the `pa_core.cli` command. Once the simulation finishes, Streamlit opens in your browser.

### Option 2 – manual launch

Run the app directly:

```bash
streamlit run dashboard/app.py
```

Provide the path to `Outputs.xlsx` in the sidebar. If the companion `Outputs.parquet` file is present, additional charts such as the funding fan become available.
The headline tab shows a risk‑return scatter while other tabs visualise cumulative funding (`Funding fan`) and return distributions (`Path dist`). Adjust parameters and reload the file to test how the risk‑return profile shifts.

### Sidebar Controls

- **Results file** – text box to locate the Excel output.
- **Months** – slider to limit the number of months shown.
- **Agents** – choose which sleeves to display.
- **Risk‑free rate** – for any excess return calculations.
- **Auto‑refresh** – polls the file every few seconds so the dashboard updates while simulations run.

Two download buttons allow you to save the headline PNG chart and the Excel file.

### Tutorial 4 – Export Charts

The CLI can create static images or PPTX packs as part of a run. Combine the following flags as needed:

```text
--png  --pdf  --pptx  --html  --gif  --alt-text "Description"
```

`--html` saves an interactive Plotly page, while `--gif` exports an animation of monthly paths.  The optional `--alt-text` flag attaches descriptive text to HTML and PPTX exports so charts remain accessible.  You can also run `scripts/visualise.py` after a simulation to generate additional charts from the saved output files.

### Tutorial 5 – Generate Custom Visualisations

Use `scripts/visualise.py` to build plots outside the dashboard. The script
reads the Excel output along with an optional `.parquet` file of raw paths and
can export any Plotly figure. Pass one of the following names to `--plot`:
`risk_return`, `fan`, `path_dist`, `corr_heatmap`, `sharpe_ladder`,
`rolling_panel` or `surface`.  Combine with `--png`, `--pdf`, `--pptx`,
`--html`, `--gif` and an optional `--alt-text` description to save images:

```bash
python scripts/visualise.py \
  --plot risk_return \
  --xlsx Outputs.xlsx \
  --png --alt-text "Risk-return chart"
```

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
3. Allocate capital to the new agent in your CSV or YAML configuration file.
4. Run the CLI again and the sleeve will automatically appear in the outputs.

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

```yaml
# config_thresholds.yaml
shortfall_green: 0.05
shortfall_amber: 0.10
```

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

After completing the tutorials you can stress‑test your assumptions by running multiple scenarios. Vary the capital weights, change the alpha streams or tweak the financing parameters in your configuration file. Re‑run the CLI and compare the resulting **ShortfallProb** and **TrackingErr** columns. Use the dashboard and export scripts to visualise how each scenario moves the portfolio relative to your targets.

### Tutorial 9 – Save Everything with Export Bundles

Use the `viz.export_bundle.save` helper to output PNG, HTML and JSON files for multiple figures at once. Pass a list of figures and a file stem:

```python
from pa_core.viz import export_bundle, risk_return, fan
figs = [risk_return.make(df_summary), fan.make(df_paths)]
export_bundle.save(figs, "plots/summary")
```

The helper writes the files under `plots/` so you can archive an entire run with one call. Combine this with the CLI export flags when you need a full set of images.

### Tutorial 10 – Explore the Chart Gallery

A Jupyter notebook `viz_gallery.ipynb` at the project root demonstrates every chart function with sample data. Launch it after installing the package in editable mode:

```bash
pip install -e .
jupyter notebook viz_gallery.ipynb
```

Adjust the parameters in the notebook to see how colours and thresholds react. The gallery is a quick way to experiment with new scenarios and visual styles without extra code.
