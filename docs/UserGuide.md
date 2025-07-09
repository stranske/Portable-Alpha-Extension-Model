# Portable Alpha-Extension Model User Guide

This guide walks through the basics of running the Monte Carlo model and explains how to explore results with the provided Streamlit dashboard.

## 1. Getting Started

1. Run `./setup.sh` once to create a virtual environment and install dependencies.
2. Execute the command‑line interface with either a CSV or YAML parameter file. For example:

```bash
python -m pa_core --params parameters.csv --index sp500tr_fred_divyield.csv
```

The simulation writes results to `Outputs.xlsx` and also saves the monthly return paths in `Outputs.parquet` for use in the dashboard.

## 2. Understanding Monte Carlo Output

Each run generates many alternate histories of index and alpha returns. The Excel file summarises annual return, volatility, Value at Risk, tracking error and breach probability for each sleeve. Review the "Inputs" sheet to confirm parameters and the "Summary" sheet to compare sleeves.

## 3. Launching the Streamlit Dashboard

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

## 4. Visualisations

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

