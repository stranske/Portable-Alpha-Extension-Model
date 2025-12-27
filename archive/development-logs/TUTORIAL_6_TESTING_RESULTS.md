# TUTORIAL 6 TESTING RESULTS - NEW USER PERSPECTIVE

## 📋 **Tutorial 6: Implement a New Agent** - Testing Attempted

### **What Tutorial 6 Covers:**
1. Create a new class under `pa_core/agents/` that subclasses `BaseAgent` and implement `monthly_returns`  
2. Register the class in `_AGENT_MAP` inside `pa_core/agents/registry.py`  
3. Allocate capital to the new agent in your CSV or YAML configuration file  
4. Run the CLI again and the new sleeve appears in the outputs

---

## 🚧 **Testing Steps and Observations:**

### **Step 1: Create Agent Class**
- Created `pa_core/agents/my_agent.py` subclassing `BaseAgent` with a dummy `monthly_returns` implementation
- ✅ File loads without syntax errors

### **Step 2: Register in Registry**
- Added entry to `_AGENT_MAP` in `pa_core/agents/registry.py`:
  ```python
  from .my_agent import MyAgent
  _AGENT_MAP["MyAgent"] = MyAgent
  ```
- ✅ Registry builds the new class without KeyError

### **Step 3: Allocate Capital via Config**
- Attempted to allocate capital by adding `my_agent_capital` in a YAML config:
  ```yaml
  N_SIMULATIONS: 50
  N_MONTHS: 12
  external_pa_capital: 0.0
  active_ext_capital: 0.0
  internal_pa_capital: 0.0
  my_agent_capital: 1000.0
  total_fund_capital: 1000.0
  ```
- ❌ **Issue**: `ModelConfig` does not recognize `my_agent_capital` field and raises validation error
- **Workaround**: Capital fields are hard-coded (`external_pa_capital`, `active_ext_capital`, `internal_pa_capital`), so new agent capital cannot be allocated without code changes

### **Step 4: Run CLI**
- Without valid capital config, CLI fails; new agent is never invoked
- Default `leftover_beta` logic assigns to `InternalBeta`, not new agent

---

## ⚠️ **User Experience Issues Identified:**

### **🔴 HIGH PRIORITY: Configuration Inflexibility**
- **Problem**: `ModelConfig` schema does not support custom agent capital parameters
- **User Impact**: Cannot allocate capital to newly registered agent via config
- **Missing**: Instructions on how to modify code to accept custom agent parameters

### **🟡 MEDIUM PRIORITY: Tutorial Assumptions**
- **Problem**: Tutorial assumes config-driven agent allocation; code requires manual schema update
- **Missing**: Guidance on extending `ModelConfig` fields and `build_from_config` logic

### **🟡 MEDIUM PRIORITY: Example Implementation**
- **Problem**: No sample code template provided for new agent implementation
- **Missing**: Example `my_agent.py` and registry snippet in tutorial

### **🟡 MEDIUM PRIORITY: Integration Demonstration**
- **Problem**: Tutorial lacks end-to-end example showing config, code, and output
- **Missing**: Demonstration of outputs.xlsx with new agent sheet and summary

---

## 🚀 **Enhancement Opportunities:**

- **Dynamic Agent Configuration**: Extend `ModelConfig` to accept a generic `agents` list with names, types, and capital allocations
- **Plugin Architecture**: Allow discovery of custom agent classes without manual registry edits
- **Parameter Sweep Integration**: Showcase agent performance across parameter sweeps for custom agents
- **Code Templates**: Provide example code snippets and config YAML in tutorial resources

---

## 🔧 **Tutorial 6 Status: NOT FULLY WORKING - Requires Code Extension**

**Core Steps**: ⚠️ Partially works (agent class + registry OK)  
**Config Allocation**: ❌ Fails due to rigid schema  
**CLI Integration**: ❌ CLI does not accept custom agent capital parameters  

**Immediate Fixes Needed:**
1. **ModelConfig Extension**: Add support for `my_agent_capital` or generic agent allocation fields  
2. **build_from_config Update**: Include new agent in factory logic based on config keys  
3. **Tutorial Update**: Provide code templates and explain code modifications

**Next**: Update codebase and tutorial to enable configuration-driven custom agent implementation.
