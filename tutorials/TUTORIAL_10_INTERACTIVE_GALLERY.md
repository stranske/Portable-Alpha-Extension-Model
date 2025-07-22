# Tutorial 10: Interactive Gallery Enhancement

**🎯 Goal**: Explore every chart function using the Jupyter notebook gallery.

**⏱️ Duration**: 20 minutes
**📋 Prerequisites**: Completion of Tutorial 9
**🛠️ Tools**: `viz_gallery.ipynb`, `export_bundle.save`, `scripts/visualise.py`

### Setup

Install Streamlit and Kaleido plus a local Chrome or Chromium browser so the dashboard and static exports work:

```bash
pip install streamlit kaleido
sudo apt-get install -y chromium-browser
```

### Step 1 – Launch the notebook

```bash
pip install -e .
jupyter notebook viz_gallery.ipynb
```

Generate a small results file so the notebook has real data:

```bash
python -m pa_core.cli --config params_template.yml \
  --index sp500tr_fred_divyield.csv --output GalleryDemo.xlsx
```

### Step 2 – Experiment with plots

Load `GalleryDemo.xlsx` (and `GalleryDemo.parquet` if available) and run the cells to create figures. The notebook demonstrates advanced plots such as `overlay`, `category_pie`, `gauge`, `moments_panel`, `scenario_slider`, `spark_matrix`, `weighted_stack`, `geo_exposure`, `seasonality_heatmap`, `beta_te_scatter`, `milestone_timeline`, `mosaic`, `metric_selector`, `boxplot`, `delta_heatmap`, `quantile_band` and `triple_scatter`.

### Step 3 – Save a gallery

Use `export_bundle.save` or `scripts/visualise.py` to batch export charts for each scenario.

---

*Tutorial 10 Enhanced: Interactive Gallery*
