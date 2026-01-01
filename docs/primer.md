# Portable Alpha Primer

This brief guide explains key terms in plain English and shows how to run the model.

## Key Terms

- **Active share** – Percentage of holdings that differ from a benchmark. Higher values mean the portfolio departs more from the index.
- **Buffer multiple** – Multiplier on volatility used to size the cash buffer for downside protection.
- **Breach probability** – Chance that returns fall below the buffer threshold during a period.
- **Tracking error (TE)** – Annualised volatility of active returns (portfolio minus benchmark).
- **Conditional Value at Risk (CVaR)** – Expected loss once losses exceed the usual Value‑at‑Risk cutoff.

## Quick CLI examples

Run a simulation and save results:

```bash
python -m pa_core.cli --config config/params_template.yml --index sp500tr_fred_divyield.csv
```

Launch the interactive dashboard:

```bash
python -m dashboard.cli
```
