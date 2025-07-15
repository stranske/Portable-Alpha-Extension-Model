# TUTORIAL 10 TESTING RESULTS - NEW USER PERSPECTIVE

## 📋 **Tutorial 10: Explore the Chart Gallery** - Testing Complete

### **What Tutorial 10 Covers:**
- A Jupyter notebook `viz_gallery.ipynb` demonstrating every chart function with sample data.
- Interactive inline display of summary and path-based charts.

---

## 🚀 **Testing Steps and Observations:**

1. Installed package in editable mode:
   ```bash
   pip install -e .
   ```
2. Launched the notebook:
   ```bash
   jupyter notebook viz_gallery.ipynb
   ```
3. Executed import cell: `import pandas as pd` and `from pa_core import viz`.
4. Ran the example summary table cell and observed correct DataFrame creation.
5. Inserted and ran new code cell to generate all chart types:
   - Summary-based: `risk_return`, `sharpe_ladder`, `rolling_panel`, `surface`.
   - Path-based: created synthetic `df_paths` and generated `corr_heatmap`, `fan`, `path_dist`.
6. Verified inline display of Plotly figures for each chart.

---

## ⚠️ **User Experience Issues Identified:**

- **No Pre-Built Data Sources**: Notebook does not include real simulation outputs or parquet examples; requires manual data creation.
- **Lack of Export Examples**: No demonstration of saving charts to files via `export_bundle` or CLI scripts.
- **Minimal Narrative Guidance**: Cells lack explanations of chart parameters, thresholds, and interpretation.
- **No Batch Workflow**: Tutorial shows one-off chart calls; no examples for looping or gallery exports.

---

## 🚀 **Enhancement Opportunities:**

- Add cells loading actual Excel/Parquet outputs from recent runs for real-world data.
- Demonstrate `export_bundle.save` or `scripts/visualise.py` within notebook for file exports.
- Include markdown commentary on each chart’s purpose, parameter use, and threshold settings.
- Provide batch generation code to loop over multiple scenarios and build a gallery.
- Showcase theme customization and threshold reload via `viz.theme.reload_theme()` and `reload_thresholds()`.

---

## ✅ **Tutorial 10 Status: PARTIALLY WORKING - Core Display Only**

**Core Chart Rendering**: ✅ Interactive inline plots for all chart types  
**Data Preloading**: ❌ No built-in data sources from simulations  
**Export Demonstrations**: ❌ Missing file export examples  
**Narrative Guidance**: ❌ Minimal explanation of charts and use cases  
**Batch Workflows**: ❌ No examples of gallery or batch exports

**Immediate Next Steps:**
1. Pre-load simulation outputs and parquet data in notebook.  
2. Demonstrate export workflows (PNG, HTML, JSON, PPTX) within notebook.  
3. Add narrative markdown cells explaining each figure.  
4. Include batch loop examples to create multi-scenario galleries.  
5. Highlight theme and threshold integrations for styling.
