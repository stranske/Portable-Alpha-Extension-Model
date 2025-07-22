# Tutorial 6: Dynamic Agent Configuration

**🎯 Goal**: Add a custom agent and allocate capital to it using the configuration system.

**⏱️ Duration**: 30 minutes
**📋 Prerequisites**: Completion of Tutorial 5
**🛠️ Tools**: `pa_core.agents`, `pa_core.config`, parameter sweep engine

### Step 1 – Create a new agent

```python
# pa_core/agents/my_agent.py
from pa_core.agents.base import BaseAgent

class MyAgent(BaseAgent):
    def monthly_returns(self, r_beta, alpha_stream, financing):
        return r_beta + alpha_stream - financing
```

Register the class in `pa_core/agents/registry.py` by adding it to `_AGENT_MAP`.

### Step 2 – Extend the configuration

Add a capital field to `ModelConfig` and handle it in `build_from_config`:

```python
class ModelConfig(BaseModel):
    my_agent_capital: float = 0.0
```

When `my_agent_capital` is positive, create an `AgentParams` entry for `MyAgent`.
Include the new field in your YAML or CSV template so the CLI knows how much to allocate.

### Step 3 – Run a parameter sweep

Create a CSV file `custom_agent_sweep.csv` with a `my_agent_capital` column listing the allocations to test.
Run the CLI in capital mode:

```bash
python -m pa_core.cli \
  --mode capital \
  --params custom_agent_sweep.csv \
  --output MyAgentSweep.xlsx
```

### Step 4 – Visualise the results

Load `MyAgentSweep.xlsx` in the dashboard or build charts manually:

```python
import pandas as pd
from pa_core.viz import risk_return

df = pd.read_excel("MyAgentSweep.xlsx", sheet_name="Summary")
fig = risk_return.make(df)
fig.write_image("plots/my_agent_sweep.png")
```

Use these outputs to compare performance across the range of capital allocations and identify optimal settings.

### Step 5 – Add custom metrics

Extend the summary table with additional calculations to analyse the new agent
across the sweep results. For example, compute a simple return‑over‑risk ratio
and regenerate the chart:

```python
import pandas as pd
from pa_core.viz import risk_return

df_summary = pd.read_excel("MyAgentSweep.xlsx", sheet_name="Summary")
df_summary["ReturnOverRisk"] = df_summary["AnnReturn"] / df_summary["AnnVol"]
fig = risk_return.make(df_summary)
fig.write_image("plots/my_agent_custom.png")
```

Update your visualisation scripts to use any custom columns when comparing
scenarios.

---

**Next Tutorial**: Advanced Theme Integration – learn how to customise the colour scheme and thresholds for your charts.

*Tutorial 6 Enhanced: Dynamic Agent Configuration*
