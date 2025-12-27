# TUTORIAL 5 TESTING RESULTS - NEW USER PERSPECTIVE

## 📋 **Tutorial 5: Generate Custom Visualisations** - Testing Complete

### **What Tutorial 5 Covers:**
✅ Use `scripts/visualise.py` to build plots outside the dashboard  
✅ Pass `--plot` names: `risk_return`, `fan`, `path_dist`, `corr_heatmap`, `sharpe_ladder`, `rolling_panel`, `surface`  
✅ Combine with flags: `--png`, `--pdf`, `--pptx`, `--html`, `--gif`, and `--alt-text` description  

---

## ✅ **What Worked Well:**

### **HTML Export for Summary-Based Plots**
- ✅ **risk_return**: Creates `plots/risk_return.html` (interactive chart)
- ✅ **corr_heatmap**: Creates `plots/corr_heatmap.html` (interactive heatmap)
- ✅ **sharpe_ladder**: Creates `plots/sharpe_ladder.html` (interactive bar chart)
- ✅ **rolling_panel**: Creates `plots/rolling_panel.html` (interactive panel)
- ✅ **surface**: Creates `plots/surface.html` (interactive surface chart)

### **Ease of Use**
- ✅ Simple CLI interface for each plot type
- ✅ Alt-text support embedded in HTML and PPTX exports
- ✅ Consistent directory structure (`plots/` folder) for all outputs

---

## ⚠️ **User Experience Issues Identified:**

### **🔴 HIGH PRIORITY: Chrome/Kaleido Dependency for Static Exports**
- **Problem**: `fig.write_image` calls require Chrome; attempts to export to PNG/PDF/PPTX via script fail or silent skip without explicit error
- **User Impact**: Users cannot generate static images or presentations without manual Chrome install
- **Missing**: Clear installation instructions or fallback methods for image exports

### **🟡 MEDIUM PRIORITY: Parquet File Requirement for Path-Based Plots**
- **Problem**: `fan` and `path_dist` require `Outputs.parquet` (converted from `AllReturns`); basic simulations do not generate this file
- **User Impact**: Running `--plot fan` or `--plot path_dist` raises `FileNotFoundError`
- **Missing**: Tutorial instructions on how to create parquet files (AllReturns conversion)

### **🟡 MEDIUM PRIORITY: GIF Export Logging and Feedback**
- **Problem**: `--gif` flag now attempts GIF creation via `write_gif` but failures only appear in logs
- **User Impact**: Users may not see warnings or know why GIF export failed
- **Missing**: Tutorial documentation on log inspection and guidance for resolving GIF issues

### **🟡 MEDIUM PRIORITY: Comprehensive Plot Type Guidance**
- **Problem**: Tutorial provides list of plot names but no descriptions or examples for each
- **Missing**: When to use each plot type and what insights they provide
- **Missing**: Best practices for combining export formats

---

## 📊 **Actual Results from Testing:**

### **Example Commands and Outputs:**
```bash
# Interactive HTML export for risk_return
python scripts/visualise.py --plot risk_return --xlsx Outputs.xlsx --html --alt-text "Risk-return chart"
→ Creates: plots/risk_return.html

# Attempt static export (PNG)
python scripts/visualise.py --plot risk_return --xlsx Outputs.xlsx --png
→ No plots/risk_return.png created (requires Chrome)

# Path-based plot error (fan) without parquet
python scripts/visualise.py --plot fan --xlsx Outputs.xlsx --html
→ FileNotFoundError: Outputs.parquet
```

### **Directory Contents (`plots/`):**
```
plots/
├── risk_return.html
├── corr_heatmap.html
├── sharpe_ladder.html
├── rolling_panel.html
├── surface.html
└── summaries from previous tutorials
```

---

## 🚀 **Enhancement Opportunities Using Parameter Sweeps:**

### **Current Tutorial Limitation:**
- Generates only single-scenario charts
- No bulk production of multi-scenario visualisations
- No systematic comparison across parameter combinations

### **Parameter Sweep Enhancement Potential:**
- **Bulk chart generation**: Loop over parameter sweep results to create 50-200 interactive charts automatically
- **Multi-scenario galleries**: Build HTML galleries or PPTX decks summarizing sweep outputs
- **Animation workflows**: Generate animated sequences across parameter steps
- **Professional reporting**: Embed charts into automated client-ready presentations

### **Enhanced Tutorial 5 Structure:**
1. **Part 1**: Single-scenario chart generation (current functionality)  
2. **Part 2**: Parameter sweep chart gallery creation (bulk iteration example)  
3. **Part 3**: Interactive HTML dashboards for sweep results  
4. **Part 4**: Automated PPTX deck generation for multi-scenario reporting

---

## ✅ **Tutorial 5 Status: PARTIALLY WORKING - Needs Dependency & Guidance Fixes**

**Core Functionality**: ✅ Interactive HTML exports work  
**Static Exports**: ❌ Fails silently due to Chrome/Kaleido dependency  
**Path-Based Plots**: ❌ Require parquet conversion not documented  
**User Experience**: ⚠️ Needs installation guidance and plot documentation  
**Parameter Sweep Ready**: 🚀 Can be enhanced with bulk visualization workflows

**Immediate Fixes Needed:**
1. **Install Chrome for Kaleido** or provide fallback dependency installation in tutorial
2. **Add Parquet Conversion Instructions** for path-based plots
3. **Document Logging Behavior** for static (PNG/PDF) and animation (GIF) exports
4. **Expand Plot Descriptions** in tutorial with use cases and examples

**Next**: Incorporate these fixes and enhancements into Tutorial 5 documentation and CodeX instructions.
