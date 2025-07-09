# Portable Alpha-Extension Model User Guide

This program simulates a portable-alpha plus active-extension strategy. Each run distributes capital across internal, external portable-alpha and active-extension sleeves and draws joint return paths. The command line prints a summary and writes an Excel workbook along with optional charts. The sections below show how to configure a run, interpret the results and visualise key metrics.


## 1. Getting Started

1. Run `./setup.sh` once to create a virtual environment and install all dependencies.
2. Launch the CLI with a CSV `--params` file or YAML `--config` file (templates live in `config/`). Supply your index returns via `--index`.

```bash
python -m pa_core --params parameters.csv --index sp500tr_fred_divyield.csv
```
The run prints a console summary and writes an Excel workbook (`Outputs.xlsx` by default). Monthly return paths are stored in `Outputs.parquet` for the dashboard.

## 2. Tutorial 1 – Configure and Run a Simulation

Edit one of the templates in `config/` or create your own CSV of parameters. Then run the CLI to generate results. Use `--output` to change the Excel filename and `--pivot` to append raw returns.

```bash
python -m pa_core \
  --config my_params.yml \
  --index sp500tr_fred_divyield.csv \
  --output Results.xlsx \
  --pivot
```

Set `--seed` for reproducible draws or `--backend cupy` if a GPU is available.

## 3. Tutorial 2 – Interpret the Excel Output

Each run prints a Rich table of headline metrics and generates many alternate histories of returns. The Excel file summarises Annual Return, Annual Volatility, Value at Risk, Tracking Error and **ShortfallProb** for each sleeve. Review the `Inputs` sheet to confirm parameters and the `Summary` sheet to compare sleeves.

## 4. Tutorial 3 – Interactive Dashboard

After producing an output file you can start an interactive dashboard to visualise the results.

### Option 1 – from the CLI

Add `--dashboard` to the `pa_core` command. Once the simulation finishes, Streamlit opens in your browser.

### Option 2 – manual launch

Run the app directly:

```bash
streamlit run dashboard/app.py
```

Provide the path to `Outputs.xlsx` in the sidebar. If the companion Parquet file is present, additional charts become available.

### Sidebar Controls

- **Results file** – text box to locate the Excel output.
- **Months** – slider to limit the number of months shown.
- **Agents** – choose which sleeves to display.
- **Risk‑free rate** – for any excess return calculations.
- **Auto‑refresh** – polls the file every few seconds so the dashboard updates while simulations run.

Two download buttons allow you to save the headline PNG chart and the Excel file.

## 5. Tutorial 4 – Exporting Charts

The CLI can create static images or PPTX packs as part of a run. Combine the following flags as needed:

```text
--png  --pdf  --pptx  --html  --gif  --alt-text "Description"
```

`--html` saves an interactive Plotly page, while `--gif` exports an animation of monthly paths.  Alt text ensures exported charts remain accessible.

## 6. Dashboard Views

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

