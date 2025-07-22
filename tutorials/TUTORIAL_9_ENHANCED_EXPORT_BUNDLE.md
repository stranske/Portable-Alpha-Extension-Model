# Tutorial 9: Enhanced Export Bundle Integration

**🎯 Goal**: Save complete sets of figures with flexible naming and format options.

**⏱️ Duration**: 15 minutes
**📋 Prerequisites**: Completion of Tutorial 8
**🛠️ Tools**: `viz.export_bundle`, CLI export flags

### Setup

Install Streamlit and Kaleido plus a local Chrome or Chromium browser so the dashboard and static exports work:

```bash
pip install streamlit kaleido
sudo apt-get install -y chromium-browser
```
When generating results for this tutorial, always pass a unique `--output` name
to the CLI so each sweep saves to a new workbook.

### Step 1 – Save multiple figures at once

```python
from pa_core.viz import export_bundle, risk_return, fan
figs = [risk_return.make(df_summary), fan.make(df_paths)]
export_bundle.save(figs, "plots/summary")
```

The helper writes PNG, HTML and JSON files by default. Use `--alt-text` with the CLI for accessible captions.
If Chrome or Kaleido are missing the PNG step logs a warning and only the HTML
and JSON files appear. Check the console output if an image is missing.

For additional formats call the matching helpers or enable the CLI flags:
`--pdf`, `--pptx` and `--gif` generate extra files for each figure when
supported.

### Step 2 – Loop over scenarios

When working with parameter sweeps, generate one bundle per scenario:

```python
for label, df in sweep_dfs.items():
    figs = [risk_return.make(df), fan.make(paths[label])]
    export_bundle.save(figs, f"plots/{label}")
```
Pass `alt_texts` to label each image and adjust the prefix to control file naming. Each call writes `prefix_1.png`, `prefix_1.html` and so on.

Combine this approach with the CLI export flags to archive an entire run automatically.

### Step 3 – Extend with custom formats

Use the dedicated helpers to generate additional file types and keep each
scenario in its own folder for easy auditing:

```python
from pathlib import Path
from pa_core.viz import pdf_export, pptx_export

for label, df in sweep_dfs.items():
    figs = [risk_return.make(df), fan.make(paths[label])]
    stem = Path(f"archive/{label}/chart")
    export_bundle.save(figs, stem)
    pdf_export.save(figs[0], stem.with_suffix(".pdf"))
    pptx_export.save(figs, stem.with_suffix(".pptx"))
```

This plugin-style approach preserves scenario labels in the file names and adds
PDF or PPTX versions alongside the standard PNG/HTML/JSON files.

---

**Next Tutorial**: Interactive Gallery Enhancement – explore every chart type in the Jupyter notebook.

*Tutorial 9 Enhanced: Export Bundle Integration*
