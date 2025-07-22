# Tutorial 9: Enhanced Export Bundle Integration

**🎯 Goal**: Save complete sets of figures with flexible naming and format options.

**⏱️ Duration**: 15 minutes
**📋 Prerequisites**: Completion of Tutorial 8
**🛠️ Tools**: `viz.export_bundle`, CLI export flags

### Step 1 – Save multiple figures at once

```python
from pa_core.viz import export_bundle, risk_return, fan
figs = [risk_return.make(df_summary), fan.make(df_paths)]
export_bundle.save(figs, "plots/summary")
```

The helper writes PNG, HTML and JSON files by default. Use `--alt-text` with the CLI for accessible captions.

### Step 2 – Loop over scenarios

When working with parameter sweeps, generate one bundle per scenario:

```python
for label, df in sweep_dfs.items():
    figs = [risk_return.make(df), fan.make(paths[label])]
    export_bundle.save(figs, f"plots/{label}")
```

Combine this approach with the CLI export flags to archive an entire run automatically.

---

**Next Tutorial**: Interactive Gallery Enhancement – explore every chart type in the Jupyter notebook.

*Tutorial 9 Enhanced: Export Bundle Integration*
