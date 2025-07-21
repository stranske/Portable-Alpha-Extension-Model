# Tutorial 4: Professional Bulk Export Workflows

**🎯 Goal**: Generate presentation-ready charts from parameter sweeps using the CLI export flags.

**⏱️ Duration**: 15-20 minutes
**📋 Prerequisites**: Completion of Tutorial 3 with working Excel results
**🛠️ Tools**: `pa_core.cli`, `scripts/visualise.py`, Chrome/Chromium + Kaleido

### Setup

Ensure Kaleido and a local Chrome or Chromium browser are installed so static exports work:

```bash
pip install kaleido
sudo apt-get install -y chromium-browser
```
If these dependencies are missing the CLI logs a warning and no image is
created, so check the console if a file does not appear. Use the
`--alt-text` flag to embed accessible descriptions in HTML and PPTX exports.

### 1. Export During a Simulation

Pass one or more export flags when running the CLI. The example below generates a PPTX deck for every scenario in a capital sweep:

```bash
python -m pa_core.cli \
  --params config/capital_mode_template.csv \
  --mode capital \
  --pptx --output CapitalSweep.xlsx
```

The command writes `CapitalSweep.xlsx` and a matching `CapitalSweep.pptx` under `plots/`. Each scenario becomes a slide with the headline risk‑return chart.

### 2. Combine Multiple Formats

Export PNG, PDF and HTML in one run:

```bash
python -m pa_core.cli \
  --params config/alpha_shares_mode_template.csv \
  --mode alpha_shares \
  --png --pdf --html --output AlphaSweep.xlsx
```

Files are stored in `plots/` with names like `summary.png` and `summary.pdf`.

> **Tip**: Include `--gif` to create an animated path visualisation when using `scripts/visualise.py`.

### 3. Post‑Run Chart Generation

Use `scripts/visualise.py` to create additional charts after a simulation completes:

```bash
python scripts/visualise.py \
  --plot rolling_panel \
  --xlsx CapitalSweep.xlsx \
  --png --alt-text "Rolling drawdown and TE"
```

Provide an accompanying Parquet file for path-based plots like `fan` or `path_dist`:

```bash
python scripts/visualise.py \
  --plot fan \
  --xlsx CapitalSweep.xlsx \
  --parquet CapitalSweep.parquet \
  --png
```

### 4. Best Practices

- Always specify `--output` so previous results are preserved.
- Combine export flags to produce all required formats in one command.
- For sweep files, check the `plots/` folder for one image per scenario when using PPTX or GIF outputs.

---

*Tutorial 4 Enhanced: Professional Bulk Export Workflows*
